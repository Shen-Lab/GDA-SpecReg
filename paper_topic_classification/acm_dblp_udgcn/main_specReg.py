# coding=utf-8
import os

from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import utils
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--UDAGCN", type=bool,default=True)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--gamma_adv", type=float, default=0.1)
parser.add_argument("--thr_smooth", type=float, default=-1)
parser.add_argument("--gamma_smooth", type=float, default=0.01)
parser.add_argument("--thr_mfr", type=float, default=-1)
parser.add_argument("--gamma_mfr", type=float, default=0.01)


args = parser.parse_args()
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim



id = "source: {}, target: {}, UDAGCN: {}, encoder_dim: {}"\
    .format(args.source, args.target, use_UDAGCN,  encoder_dim)

print(id)



rate = 0.0
random.seed(np.random.randint(1000))
np.random.seed(np.random.randint(1000))
torch.manual_seed(200)
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
eival_src, eivec_src = dataset.eival.cuda(), dataset.eivec.cuda()
print(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
eival_tgt, eivec_tgt = dataset.eival.cuda(), dataset.eivec.cuda()
print(target_data)


source_data = source_data.to(device)
target_data = target_data.to(device)



class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)


encoder = GNN(type="gcn").to(device)
if use_UDAGCN:
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)


cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


att_model = Attention(encoder_dim).cuda()

models = [encoder, cls_model, domain_model]
if use_UDAGCN:
    models.extend([ppmi_encoder, att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        outputs = att_model([gcn_output, ppmi_output])
        return outputs
    else:
        return gcn_output

def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy




critic = nn.Sequential( nn.Linear(args.encoder_dim, args.encoder_dim), nn.ReLU(), nn.Linear(args.encoder_dim, args.encoder_dim), nn.ReLU(), nn.Linear(args.encoder_dim, 1) ).to('cuda')
optimizer_critic = torch.optim.Adam(critic.parameters(), 1e-4)
relu = nn.ReLU()


epochs = 200
def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    if use_UDAGCN:
        _x_src, _x_tgt = encoded_source.detach(), encoded_target.detach()
        for _ in range(5):
            optimizer_critic.zero_grad()
            loss_1 = critic(_x_src).mean() - critic(_x_tgt).mean()
            loss_2 = utils.calculate_gradient_penalty(critic, _x_src, _x_tgt)
            loss_adv = - loss_1 + 10 * loss_2
            loss_adv.backward(); optimizer_critic.step()

        loss_grl = critic(encoded_source).mean() - critic(encoded_target).mean()
        loss = cls_loss + loss_grl * args.gamma_adv

        x_src = torch.einsum('nm,md->nd', eivec_src, encoded_source)
        x_tgt = torch.einsum('nm,md->nd', eivec_tgt, encoded_target)
        if args.thr_smooth > 0:
            delta_src = (x_src[:-1] - x_src[1:]).abs()
            delta_tgt = (x_tgt[:-1] - x_tgt[1:]).abs()
            loss = loss + ( relu(delta_src - args.thr_smooth).mean() + relu(delta_tgt - args.thr_smooth).mean() ) * args.gamma_smooth
        if args.thr_mfr > 0:
            loss = loss + ( relu(x_src.abs() - args.thr_mfr).mean() + relu(x_tgt.abs() - args.thr_mfr).mean() ) * args.gamma_mfr

        # use target classifier loss:
        target_logits = cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy* (epoch / epochs * 0.01)

    else:
        loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    target_correct = test(target_data, "target")
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch, best_source_acc, best_target_acc)

print(line)
print(best_target_acc.item() * 100)


