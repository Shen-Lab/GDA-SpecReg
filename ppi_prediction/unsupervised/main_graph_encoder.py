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
loss_func = nn.BCELoss()

# optimizer
optimizer_seq = torch.optim.Adam(seq_encoder.parameters(), args.learning_rate)
optimizer_graph = torch.optim.Adam(graph_encoder.parameters(), args.learning_rate)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), args.learning_rate)


def train():
    loss_total = 0
    seq_encoder.train()
    graph_encoder.train()
    classifier.train()
    for data_src, data_tgt in dataloader:
        # skip subgraph without labels
        if data_src.edge_index_label.shape[0] == 0:
            continue

        optimizer_seq.zero_grad()
        optimizer_graph.zero_grad()
        optimizer_classifier.zero_grad()

        # forward propagation
        seq_src, seq_tgt = [G_src.nodes[n]['sequence'] for n in data_src.x.numpy().tolist()], [G_tgt.nodes[n]['sequence'] for n in data_tgt.x.numpy().tolist()]
        data_src, data_tgt = data_src.to('cuda'), data_tgt.to('cuda')
        x_src, x_tgt = seq_encoder(seq_src), seq_encoder(seq_tgt)
        x_src, x_tgt = graph_encoder(x_src, data_src.edge_index, data_src.edge_attr), graph_encoder(x_tgt, data_tgt.edge_index, data_tgt.edge_attr)

        # calculate loss
        edge_index_neg = data_src.edge_index_label_neg
        edge_index_label = torch.cat([data_src.edge_index_label, edge_index_neg], dim=1)
        label = torch.cat([data_src.edge_attr_label, torch.zeros(edge_index_neg.shape[1], 2, dtype=torch.float32).to('cuda')], axis=0)
        pred = classifier( x_src[edge_index_label[0]] + x_src[edge_index_label[1]] )
        loss = ( loss_func(pred[label==0], label[label==0]) + loss_func(pred[label==1], label[label==1]) ) / 2

        if not data_tgt.edge_index_label.shape[0] == 0:
            edge_index_label = data_tgt.edge_index_label
            pred = classifier( x_tgt[edge_index_label[0]] + x_tgt[edge_index_label[1]] )
            label = data_tgt.edge_attr_label
            loss = loss + loss_func(pred, label)

        loss_total += loss.item()

        # backward propagation
        loss.backward()
        optimizer_seq.step()
        optimizer_graph.step()
        optimizer_classifier.step()
    return loss_total / len(dataloader)


# training and evaluation
metrics = utils.evaluate(G_src, G_tgt, seq_encoder, graph_encoder, classifier)
print('initialization validation roc/ap', metrics[0], metrics[1], 'test roc/ap coexpression', metrics[2], metrics[3], 'experiments', metrics[4], metrics[5])
auroc_max, metrics_best = np.mean(metrics[:2]), metrics
# for epoch in tqdm(range(args.epoch_num)):
for epoch in range(args.epoch_num):
    loss_total = train()
    metrics = utils.evaluate(G_src, G_tgt, seq_encoder, graph_encoder, classifier)
    print('epoch', epoch, 'training loss', loss_total, 'validation roc/ap', metrics[0], metrics[1], 'test roc/ap coexpression', metrics[2], metrics[3], 'experiments', metrics[4], metrics[5])

    # store best validated performance
    if auroc_max < np.mean(metrics[:2]):
        auroc_max, metrics_best = np.mean(metrics[:2]), metrics

print('best validation roc/ap', metrics_best[0], metrics_best[1], 'test roc/ap coexpression', metrics_best[2], metrics_best[3], 'experiments', metrics_best[4], metrics_best[5])

