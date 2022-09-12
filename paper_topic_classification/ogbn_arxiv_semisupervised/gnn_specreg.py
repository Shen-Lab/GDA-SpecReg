import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

import numpy as np
from sklearn.decomposition import TruncatedSVD
import pdb


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, critic):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            if i == len(self.convs) - 2: x_feat = x
            elif i == 0: x0_feat = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        x_proj = critic(x_feat)
        return x.log_softmax(dim=-1), x_proj, x_feat, x0_feat


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def calculate_gradient_penalty(critic, x):
    # interpolation
    x = x.requires_grad_(True)

    x_out = critic(x)
    grad_out = torch.ones(x_out.shape, requires_grad=False).to(x_out.device)

    # Get gradient w.r.t. x
    grad = torch.autograd.grad(outputs=x_out,
                               inputs=x,
                               grad_outputs=grad_out,
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True,)[0]
    grad = grad.view(grad.shape[0], -1)
    grad_penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
    return grad_penalty


def get_laplacian_evd(adj):
    # calculate laplacian matrix
    adj.setdiag(-adj.sum(axis=1))
    adj = -adj

    svd = TruncatedSVD(n_components=100, n_iter=20, random_state=42)
    svd.fit(adj)

    eival = torch.tensor( svd.explained_variance_ ** 0.5, dtype=torch.float32 ).to('cuda')
    eivec = torch.tensor( svd.components_, dtype=torch.float32 ).to('cuda')
    return eival, eivec


relu = torch.nn.ReLU()
def spectral_regularization_smooth(x, x0, eivec, eival, thres):
    x_out = torch.einsum('nm,md->nd', eivec, x)
    x0_out = torch.einsum('nm,md->nd', eivec, x0)

    delta = ( x_out[:-1].t() * eival[:-1] - x_out[1:].t() * eival[1:] ).t().abs()
    delta0 = ( x0_out[:-1].t() * eival[:-1] - x0_out[1:].t() * eival[1:] ).t().abs()

    loss = relu(delta - thres * delta0)[(eival[:-1]-eival[1:])>1e-2].mean()
    return loss


def spectral_regularization_lowpass(x, x0, eivec, eival, thres):
    x_out = torch.einsum('nm,md->nd', eivec, x)
    x0_out = torch.einsum('nm,md->nd', eivec, x0)

    loss = relu(x_out.abs() - thres * x0_out.abs())[x0_out.abs()>1e-2].mean()
    return loss


def train(model, data, train_idx, val_idx, idx_semi, optimizer, critic, optimizer_c, gamma, eival, eivec, gamma_ss, thres_ss, gamma_mfr, thres_mfr):
    model.train()

    for _ in range(5):
        optimizer_c.zero_grad()
        _, x_proj, x_feat, _ = model(data.x, data.adj_t, critic)
        loss = - x_proj[train_idx].mean() + x_proj[val_idx].mean() + 10 * calculate_gradient_penalty(critic, x_feat[train_idx].detach())
        loss.backward()
        optimizer_c.step()

    optimizer.zero_grad()
    out, x_proj, x_feat, x0_feat = model(data.x, data.adj_t, critic)
    out = out[torch.cat([train_idx, val_idx.cuda()[idx_semi]])]
    loss = F.nll_loss(out, data.y.squeeze(1)[torch.cat([train_idx, val_idx.cuda()[idx_semi]])])
    loss_c = x_proj[train_idx].mean() - x_proj[val_idx].mean()

    loss_reg = 0
    if gamma_ss > 0: loss_reg += gamma_ss * spectral_regularization_smooth(x_feat, x0_feat, eivec, eival, thres_ss)
    if gamma_mfr > 0: loss_reg += gamma_mfr * spectral_regularization_lowpass(x_feat, x0_feat, eivec, eival, thres_mfr)

    (loss + gamma * loss_c + loss_reg).backward()
    optimizer.step()

    return loss.item(), loss_c.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, critic):
    model.eval()

    out, _, _, _ = model(data.x, data.adj_t, critic)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--label_rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gamma_ss', type=float, default=0.1)
    parser.add_argument('--thres_ss', type=float, default=0.1)
    parser.add_argument('--gamma_mfr', type=float, default=0.1)
    parser.add_argument('--thres_mfr', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)


    adj = data.adj_t.to_scipy().tocsr()
    eival, eivec = get_laplacian_evd(adj)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):

        # val label idx
        valid_idx = split_idx['valid'].to(device)
        num_valid = valid_idx.shape[0]
        idx_semi = np.random.choice(num_valid, int(num_valid*0.05), replace=False)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        critic = nn.Sequential( nn.Linear(args.hidden_channels, args.hidden_channels), nn.ReLU(), nn.Linear(args.hidden_channels, args.hidden_channels), nn.ReLU(), nn.Linear(args.hidden_channels, 1) ).to(device)
        optimizer_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

        num_train = train_idx.shape[0]
        if args.label_rate < 1: idx_rand = np.random.choice(num_train, int(num_train*args.label_rate), replace=False)
        else: idx_rand = np.arange(num_train)

        for epoch in range(1, 1 + args.epochs):
            loss, loss_c = train(model, data, train_idx[idx_rand], split_idx['valid'], idx_semi, optimizer, critic, optimizer_c, args.gamma, eival, eivec, args.gamma_ss, args.thres_ss, args.gamma_mfr, args.thres_mfr)
            result = test(model, data, split_idx, evaluator, critic)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Loss Critic: {loss_c:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
