import argparse
from tqdm import tqdm
import utils
import nets
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tgeom
from sklearn.metrics import roc_auc_score, average_precision_score
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--src_species', type=str, default='human')
parser.add_argument('--tgt_species', type=str, default='yeast')
parser.add_argument('--randwalk_step', type=int, default=128)
parser.add_argument('--seq_encoder', type=str, default='transformer')
parser.add_argument('--transformer_bucket_size', type=int, default=32)
parser.add_argument('--hrnn_kmer', type=int, default=50)
parser.add_argument('--graph_encoder', type=str, default='gin')
parser.add_argument('--layer_num', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gat_attn_head', type=int, default=8)
parser.add_argument('--gin_mlp_layer', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epoch_num', type=int, default=500)
parser.add_argument('--suffix', type=str, default='1')
parser.add_argument('--adv_train_gamma', type=float, default=0.001)
parser.add_argument('--spectral_reg_smooth_gamma', type=float, default=0.1)
parser.add_argument('--spectral_reg_lowpass_gamma', type=float, default=0.1)
parser.add_argument('--spectral_reg_gamma2', type=float, default=0.1)
args = parser.parse_args()
print(args)


G_src = utils.read_ppi(args.src_species)
G_tgt = utils.read_ppi(args.tgt_species, train_ratio=0)

# dataloader
dataset = utils.dataset_pairwise(G_src, G_tgt, randwalk_step=args.randwalk_step)
dataloader = tgeom.loader.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

# model
if args.seq_encoder == 'transformer':
    seq_encoder = nets.transformer(hidden_dim=args.hidden_dim, bucket_size=args.transformer_bucket_size).to('cuda')
elif args.seq_encoder == 'hrnn':
    seq_encoder = nets.hrnn(hidden_dim=args.hidden_dim, kmer=args.hrnn_kmer).to('cuda')
if args.graph_encoder == 'gat':
    graph_encoder = nets.gat(layer_num=args.layer_num, hidden_dim=args.hidden_dim, attn_head=args.gat_attn_head).to('cuda')
elif args.graph_encoder == 'gin':
    graph_encoder = nets.gin(layer_num=args.layer_num, hidden_dim=args.hidden_dim, mlp_layer=args.gin_mlp_layer).to('cuda')
classifier = nn.Sequential( nn.Linear(args.hidden_dim, 2), nn.Sigmoid() ).to('cuda')
critic = nn.Sequential( nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(), nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(), nn.Linear(args.hidden_dim, 1) ).to('cuda')
loss_func = nn.BCELoss()

# optimizer
optimizer_seq = torch.optim.Adam(seq_encoder.parameters(), args.learning_rate)
optimizer_graph = torch.optim.Adam(graph_encoder.parameters(), args.learning_rate)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), args.learning_rate)
optimizer_critic = torch.optim.Adam(critic.parameters(), args.learning_rate)


def train():
    loss_total = [0, 0, 0, 0, 0, 0]
    seq_encoder.train(); graph_encoder.train(); classifier.train(); critic.train()
    for data_src, data_tgt in dataloader:
        # skip subgraph without labels
        if data_src.edge_index_label.shape[0] == 0:
            continue

        # forward propagation
        seq_src, seq_tgt = [G_src.nodes[n]['sequence'] for n in data_src.x.numpy().tolist()], [G_tgt.nodes[n]['sequence'] for n in data_tgt.x.numpy().tolist()]
        data_src, data_tgt = data_src.to('cuda'), data_tgt.to('cuda')
        x_src, x_tgt = seq_encoder(seq_src), seq_encoder(seq_tgt)
        x_src, x_tgt = graph_encoder(x_src, data_src.edge_index, data_src.edge_attr), graph_encoder(x_tgt, data_tgt.edge_index, data_tgt.edge_attr)

        # 1. critic training, hyper-parameter reference: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17155, https://arxiv.org/abs/1704.00028v2
        _x_src, _x_tgt = x_src.detach(), x_tgt.detach()
        for _ in range(5):
            optimizer_critic.zero_grad()
            loss_1 = critic(_x_src).mean() - critic(_x_tgt).mean()
            loss_2 = utils.calculate_gradient_penalty(critic, _x_src, _x_tgt)
            loss = - loss_1 + 10 * loss_2
            loss.backward(); optimizer_critic.step()
            loss_total[2] -= loss_1.item(); loss_total[3] += loss_2.item()

        # 2. discriminator training
        optimizer_seq.zero_grad(); optimizer_graph.zero_grad(); optimizer_classifier.zero_grad()

        # distribution matching
        loss_1 = critic(x_src).mean() - critic(x_tgt).mean()

        # spectral regularization
        if args.spectral_reg_smooth_gamma > 0: loss_2 = utils.spectral_regularization_smooth(graph_encoder, _x_src, data_src.edge_index, data_src.edge_attr, _x_tgt, data_tgt.edge_index, data_tgt.edge_attr, args.spectral_reg_smooth_gamma, args.spectral_reg_gamma2)
        else: loss_2 = torch.zeros(1).to('cuda')

        if args.spectral_reg_lowpass_gamma > 0:
            ### try: loss_3 = utils.spectral_regularization_lowpass(graph_encoder, _x_src, data_src.edge_index, data_src.edge_attr, _x_tgt, data_tgt.edge_index, data_tgt.edge_attr, args.spectral_reg_lowpass_gamma) * args.spectral_reg_gamma2
            ### except: loss_3 = torch.zeros(1).to('cuda')
            loss_3 = utils.spectral_regularization_lowpass(graph_encoder, _x_src, data_src.edge_index, data_src.edge_attr, _x_tgt, data_tgt.edge_index, data_tgt.edge_attr, args.spectral_reg_lowpass_gamma) * args.spectral_reg_gamma2
        else: loss_3 = torch.zeros(1).to('cuda')

        # supervised loss
        edge_index_neg = data_src.edge_index_label_neg
        edge_index_label = torch.cat([data_src.edge_index_label, edge_index_neg], dim=1)
        label = torch.cat([data_src.edge_attr_label, torch.zeros(edge_index_neg.shape[1], 2, dtype=torch.float32).to('cuda')], axis=0)
        pred = classifier( x_src[edge_index_label[0]] + x_src[edge_index_label[1]] )
        loss_4 = ( loss_func(pred[label==0], label[label==0]) + loss_func(pred[label==1], label[label==1]) ) / 2

        loss = loss_4 + args.adv_train_gamma * loss_1 + loss_2 + loss_3
        loss_total[0] += loss_4.item(); loss_total[1] += loss_1.item(); loss_total[4] += loss_2.item(); loss_total[5] += loss_3.item()

        # backward propagation
        loss.backward(); optimizer_seq.step(); optimizer_graph.step(); optimizer_classifier.step()

    loss_total[0] /= len(dataloader); loss_total[1] /= len(dataloader); loss_total[2] /= (len(dataloader) * 5); loss_total[3] /= (len(dataloader) * 5); loss_total[4] /= len(dataloader); loss_total[5] /= len(dataloader)
    return loss_total


# training and evaluation
metrics = utils.evaluate(G_src, G_tgt, seq_encoder, graph_encoder, classifier)
print('initialization validation roc/ap', metrics[0], metrics[1], 'test roc/ap coexpression', metrics[2], metrics[3], 'experiments', metrics[4], metrics[5])
auroc_max, metrics_best = np.mean(metrics[:2]), metrics
# for epoch in tqdm(range(args.epoch_num)):
for epoch in range(args.epoch_num):
    loss_total = train()
    metrics = utils.evaluate(G_src, G_tgt, seq_encoder, graph_encoder, classifier)
    print('epoch', epoch, 'training loss', loss_total[0], loss_total[1], loss_total[2], loss_total[3], loss_total[4], loss_total[5], 'validation roc/ap', metrics[0], metrics[1], 'test roc/ap coexpression', metrics[2], metrics[3], 'experiments', metrics[4], metrics[5])

    # store best validated performance
    if auroc_max < np.mean(metrics[:2]):
        auroc_max, metrics_best = np.mean(metrics[:2]), metrics

print('best validation roc/ap', metrics_best[0], metrics_best[1], 'test roc/ap coexpression', metrics_best[2], metrics_best[3], 'experiments', metrics_best[4], metrics_best[5])

