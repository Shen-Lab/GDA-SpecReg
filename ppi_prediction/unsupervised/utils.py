import networkx as nx
import numpy as np
import torch
import torch_geometric as tgeom
import random
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import pdb


def read_ppi(species='human', train_ratio=0.8): # species in {human, yeast, mouse, fruit_fly, zebrafish, nematode}
    species_idx_dict = {'human':'9606', 'yeast':'4932', 'mouse':'10090', 'fruit_fly':'7227', 'zebrafish':'7955', 'nematode':'6239'}
    G = nx.Graph()

    with open('../../../data/ppi/'+species_idx_dict[species]+'.protein.sequences.v11.5.fa', 'r') as f:
        data = f.read().split('\n')[:-1]

    # protein sequence mapping
    prot_seq_dict = {}
    prot_id, seq = '', ''
    for d in data:
        if '>' in d:
            if not prot_id == '':
                seq.upper()
                prot_seq_dict[prot_id] = seq
            prot_id = d.split('.')[1]
            seq = ''
        else:
            seq += d

    # get node, edge
    node_list, edge_list, edge_attr_list, label_list = [], [], [], []
    label_mask = []
    with open('../../../data/ppi/'+species_idx_dict[species]+'.protein.links.full.v11.5.txt', 'r') as f:
        data = f.read().split('\n')[1:-1]
    for d in data:
        d = d.split()

        edge_attr = np.array([float(d[2]), float(d[4]), float(d[5])], dtype=np.float32) / 1000
        # edge_attr[edge_attr<0.4] = 0
        label = np.zeros(2, dtype=np.float32)
        if int(d[7]) >= 700: label[0] = 1
        if int(d[9]) >= 700: label[1] = 1
        if ((edge_attr>=0.4).sum() == 0) and (label.sum() == 0): continue

        prot1, prot2 = d[0].split('.')[1], d[1].split('.')[1]
        node_list += [prot1, prot2]
        edge_list.append([prot1, prot2])
        edge_attr_list.append(edge_attr)
        label_list.append(label)
        # label mask
        label_mask.append(1) if label.sum() > 0 else label_mask.append(0)

    # use 20% labels as validation
    label_train_val = ['val' if n < len(label_mask)*(1-train_ratio) else 'train' for n in range(len(label_mask))]
    random.shuffle(label_train_val)
    label_train_val_2, count = [], 0
    for l in label_mask:
        if l == 0: label_train_val_2.append('unknown')
        if l == 1: label_train_val_2.append(label_train_val[count]); count += 1

    # build graph
    node_list = list(set(node_list))
    G = nx.Graph()

    prot_idx_dict = {node:n for n,node in enumerate(node_list)}
    for node in node_list:
        G.add_node(prot_idx_dict[node], protein_id=node, sequence=prot_seq_dict[node])
    # add edge
    for n, (prot1, prot2) in enumerate(edge_list):
        G.add_edge(prot_idx_dict[prot1], prot_idx_dict[prot2], edge_attr=edge_attr_list[n], label=label_list[n], train_val=label_train_val_2[n])

    # negative sample for validation
    edge_index_label = torch.tensor([[n1, n2] for n1, n2 in G.edges if not G.edges[n1, n2]['train_val'] == 'unknown'], dtype=torch.int64).t()
    edge_index_label_val = torch.tensor([[n1, n2] for n1, n2 in G.edges if G.edges[n1, n2]['train_val'] == 'val'], dtype=torch.int64).t()
    edge_index_label_neg = tgeom.utils.negative_sampling(edge_index=edge_index_label, num_nodes=len(G), num_neg_samples=edge_index_label_val.shape[1]*19).t().numpy().tolist()
    for n1, n2 in edge_index_label_neg:
        if (n1, n2) in G.edges(): G.edges[n1, n2]['train_val'] = 'val'
        else: G.add_edge(n1, n2, edge_attr=np.zeros(3, dtype=np.float32), label=np.zeros(2, dtype=np.float32), train_val='val')

    label_list = np.array(label_list)
    print(species, 'node number', len(G.nodes()), 'edge number', len(G.edges()), 'coexpression', int(label_list[:,0].sum()), 'experiments', int(label_list[:,1].sum()))
    return G


class dataset(tgeom.data.InMemoryDataset):
    def __init__(self, G=None, randwalk_step=128):
        super().__init__()
        self.G = G
        self.randwalk_step = randwalk_step

    def len(self):
        return len(self.G) // self.randwalk_step

    def get_randwalk(self, G):
        node_list = [np.random.choice(G.nodes(), 1)[0]]

        # random walk sample subgraph
        node_num, count = 1, 0
        while node_num < self.randwalk_step:
            node_last = node_list[-1]
            node_neighbor = [node for node in G.neighbors(node_last) if (G.edges[node_last,node]['edge_attr']>=0.4).sum() > 0]
            if len(node_neighbor) == 0: # isolated node
                node_list.append( np.random.choice(G.nodes(), 1)[0] )
            else:
                node_list.append( np.random.choice(node_neighbor, 1)[0] )

            # avoid isolated subgraph
            node_num_update = len(set(node_list))
            if node_num_update == node_num:
                count += 1
                if count >= 20:
                    node_list.append( np.random.choice(G.nodes(), 1)[0] )
                    count = 0
                node_num = len(set(node_list))
            else:
                count = 0
                node_num = node_num_update

        # extract subgraph
        node_list = list(set(node_list))
        x = torch.tensor(node_list)
        x_idx = torch.zeros(x.max()+1, dtype=torch.int64)
        x_idx[x] = torch.arange(x.shape[0])
        G_sub = G.subgraph(node_list)

        edge_index, edge_attr = [], []
        edge_index_label, edge_attr_label = [], []
        edge_index_label_all = []
        for n1, n2 in G_sub.edges():
            if (G_sub.edges[n1,n2]['edge_attr']>=0.4).sum() > 0:
                edge_index.append([x_idx[n1],x_idx[n2]]); edge_attr.append(G_sub.edges[n1,n2]['edge_attr'])
                edge_index.append([x_idx[n2],x_idx[n1]]); edge_attr.append(G_sub.edges[n2,n1]['edge_attr'])
            if G_sub.edges[n1,n2]['train_val'] == 'train': edge_index_label.append([x_idx[n1],x_idx[n2]]); edge_attr_label.append(G_sub.edges[n1,n2]['label'])
            if not G_sub.edges[n1,n2]['train_val'] == 'unknown': edge_index_label_all.append([x_idx[n1],x_idx[n2]])

        edge_index = torch.tensor(edge_index).t()
        edge_attr = torch.tensor(edge_attr)
        edge_index_label = torch.tensor(edge_index_label).t()
        edge_attr_label = torch.tensor(edge_attr_label)
        edge_index_label_all = torch.tensor(edge_index_label_all).t()

        # negative sample
        edge_index_label_neg = tgeom.utils.negative_sampling(edge_index=edge_index_label_all, num_nodes=len(G_sub))

        data = tgeom.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_index_label=edge_index_label, edge_attr_label=edge_attr_label, edge_index_label_neg=edge_index_label_neg)
        return data

    def get(self, idx):
        data = self.get_randwalk(self.G)
        return data


class dataset_pairwise(dataset):
    def __init__(self, G1=None, G2=None, randwalk_step=128):
        super().__init__()
        self.G1, self.G2 = G1, G2
        self.randwalk_step = randwalk_step

    def len(self):
        return len(self.G1) // self.randwalk_step

    def get(self, idx):
        data1, data2 = self.get_randwalk(self.G1), self.get_randwalk(self.G2)
        return data1, data2


def evaluate(G_src, G_tgt, seq_encoder, graph_encoder, classifier, use_gnn=True):
    metrics = []
    seq_encoder.eval()
    graph_encoder.eval()
    classifier.eval()

    # validation on source graph
    # sequence encoding
    seq_list = [G_src.nodes[n]['sequence'] for n in range(len(G_src))]
    batch_size = 512
    seq_list = [seq_list[n:n+batch_size] for n in range(0, len(seq_list), batch_size)]
    emb = []
    for seq in seq_list:
        with torch.no_grad():
            emb.append(seq_encoder(seq))
    emb = torch.cat(emb, dim=0)

    # graph encoding
    edge_index, edge_attr = [], []
    edge_index_label, edge_attr_label = [], []
    for n1, n2 in G_src.edges():
        if (G_src.edges[n1, n2]['edge_attr']>=0.4).sum() > 0:
            edge_index.append([n1, n2]); edge_attr.append(G_src.edges[n1, n2]['edge_attr'])
            edge_index.append([n2, n1]); edge_attr.append(G_src.edges[n2, n1]['edge_attr'])
        if G_src.edges[n1, n2]['train_val'] == 'val': edge_index_label.append([n1, n2]); edge_attr_label.append(G_src.edges[n1, n2]['label'])
    edge_index = torch.tensor(edge_index).t().to('cuda')
    edge_attr = torch.tensor(edge_attr).to('cuda')
    edge_index_label, edge_attr_label = torch.tensor(edge_index_label).t().to('cuda'), torch.tensor(edge_attr_label).to('cuda')
    if use_gnn:
        with torch.no_grad():
            emb = graph_encoder(emb, edge_index, edge_attr)

    # evaluate auroc and auprc with real labels
    label = edge_attr_label.to('cpu').numpy().reshape(-1)
    with torch.no_grad():
        pred = classifier( emb[edge_index_label[0]] + emb[edge_index_label[1]] ).to('cpu').numpy().reshape(-1)
    metrics += [roc_auc_score(label, pred), average_precision_score(label, pred)]

    # evaluate auroc and auprc with real labels
    label = edge_attr_label.to('cpu').numpy()
    with torch.no_grad():
        pred = classifier( emb[edge_index_label[0]] + emb[edge_index_label[1]] ).to('cpu').numpy()
    metrics_2 = [roc_auc_score(label[:,0], pred[:,0])/2 + average_precision_score(label[:,0], pred[:,0])/2, roc_auc_score(label[:,1], pred[:,1])/2 + average_precision_score(label[:,1], pred[:,1])/2]

    # test on target graph
    # sequence encoding
    seq_list = [G_tgt.nodes[n]['sequence'] for n in range(len(G_tgt))]
    batch_size = 512
    seq_list = [seq_list[n:n+batch_size] for n in range(0, len(seq_list), batch_size)]
    emb = []
    for seq in seq_list:
        with torch.no_grad():
            emb.append(seq_encoder(seq))
    emb = torch.cat(emb, dim=0)

    # graph encoding
    edge_index, edge_attr = [], []
    edge_index_label, edge_attr_label = [], []
    edge_index_label_val, edge_attr_label_val = [], []
    for n1, n2 in G_tgt.edges():
        if (G_tgt.edges[n1, n2]['edge_attr']>=0.4).sum() > 0:
            edge_index.append([n1, n2]); edge_attr.append(G_tgt.edges[n1, n2]['edge_attr'])
            edge_index.append([n2, n1]); edge_attr.append(G_tgt.edges[n2, n1]['edge_attr'])
        if G_tgt.edges[n1, n2]['label'].sum() > 0: edge_index_label.append([n1, n2]); edge_attr_label.append(G_tgt.edges[n1, n2]['label'])
        if G_tgt.edges[n1, n2]['label'].sum() > 0 and G_tgt.edges[n1, n2]['train_val'] == 'val': edge_index_label_val.append([n1, n2]); edge_attr_label_val.append(G_tgt.edges[n1, n2]['label'])
    edge_index = torch.tensor(edge_index).t().to('cuda')
    edge_attr = torch.tensor(edge_attr).to('cuda')
    edge_index_label = torch.tensor(edge_index_label).t().to('cuda')
    edge_attr_label = torch.tensor(edge_attr_label).to('cuda')
    edge_index_label_val = torch.tensor(edge_index_label_val).t().to('cuda')
    edge_attr_label_val = torch.tensor(edge_attr_label_val).to('cuda')
    if use_gnn:
        with torch.no_grad():
            emb = graph_encoder(emb, edge_index, edge_attr)

    # evaluate auroc and auprc with real labels
    edge_index_pos_all = edge_index_label[:, edge_attr_label[:,0]==1]
    edge_index_pos = edge_index_label_val[:, edge_attr_label_val[:,0]==1]
    edge_index_neg = tgeom.utils.negative_sampling(edge_index=edge_index_pos_all, num_nodes=len(G_tgt), num_neg_samples=edge_index_pos.shape[1]*19).to('cuda')
    label = torch.cat([torch.ones(edge_index_pos.shape[1]), torch.zeros(edge_index_neg.shape[1])]).to('cpu').numpy()
    edge_index_pos_neg = torch.cat([edge_index_pos, edge_index_neg], dim=1)
    with torch.no_grad():
        pred = classifier( emb[edge_index_pos_neg[0]] + emb[edge_index_pos_neg[1]] )[:,0].to('cpu').numpy()
    metrics += [roc_auc_score(label, pred), average_precision_score(label, pred)]

    edge_index_pos_all = edge_index_label[:, edge_attr_label[:,1]==1]
    edge_index_pos = edge_index_label_val[:, edge_attr_label_val[:,1]==1]
    edge_index_neg = tgeom.utils.negative_sampling(edge_index=edge_index_pos_all, num_nodes=len(G_tgt), num_neg_samples=edge_index_pos.shape[1]*19).to('cuda')
    label = torch.cat([torch.ones(edge_index_pos.shape[1]), torch.zeros(edge_index_neg.shape[1])]).to('cpu').numpy()
    edge_index_pos_neg = torch.cat([edge_index_pos, edge_index_neg], dim=1)
    with torch.no_grad():
        pred = classifier( emb[edge_index_pos_neg[0]] + emb[edge_index_pos_neg[1]] )[:,1].to('cpu').numpy()
    metrics += [roc_auc_score(label, pred), average_precision_score(label, pred)]

    metrics += metrics_2
    return metrics


def calculate_gradient_penalty(critic, x_src, x_tgt):
    # interpolation
    alpha = torch.randn((x_src.shape[0], 1)).to(x_src.device)
    x = (alpha * x_src + ((1 - alpha) * x_tgt)).requires_grad_(True)

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


def get_laplacian_evd(edge_index, node_num):
    edge_index, edge_weight = tgeom.utils.get_laplacian(edge_index, num_nodes=node_num)
    edge_index, edge_weight = edge_index.to('cpu').numpy(), edge_weight.to('cpu').numpy()
    adj = np.zeros((node_num, node_num))
    adj[edge_index[0,:], edge_index[1,:]] = edge_weight
    pca = PCA()
    pca.fit(adj)

    eival = torch.tensor( pca.explained_variance_ ** 0.5, dtype=torch.float32 ).to('cuda')
    eivec = torch.tensor( pca.components_, dtype=torch.float32 ).to('cuda')
    return eival, eivec


relu = torch.nn.ReLU()
def spectral_regularization_smooth(graph_encoder, x_src, edge_index_src, edge_attr_src, x_tgt, edge_index_tgt, edge_attr_tgt, gamma=1, gamma2=1):
    node_num = x_src.shape[0]
    loss = 0

    try: eival, eivec = get_laplacian_evd(edge_index_src, node_num)
    except: return torch.zeros(1).cuda()
    x_out = graph_encoder(x_src, edge_index_src, edge_attr_src)
    # get spectral response
    x = torch.einsum('nm,md->nd', eivec, x_src)
    x_out = torch.einsum('nm,md->nd', eivec, x_out)
    # spectral smoothness
    delta = ( x[:-1].t() * eival[:-1] - x[1:].t() * eival[1:] ).t().abs()
    out_delta = (x_out[:-1] - x_out[1:]).abs()
    loss += ( relu(out_delta - gamma * delta).mean() * gamma2 )

    try: eival, eivec = get_laplacian_evd(edge_index_tgt, node_num)
    except: return torch.zeros(1).cuda()
    x_out = graph_encoder(x_tgt, edge_index_tgt, edge_attr_tgt)
    # get spectral response
    x = torch.einsum('nm,md->nd', eivec, x_tgt)
    x_out = torch.einsum('nm,md->nd', eivec, x_out)
    # spectral smoothness
    delta = ( x[:-1].t() * eival[:-1] - x[1:].t() * eival[1:] ).t().abs()
    out_delta = (x_out[:-1] - x_out[1:]).abs()
    loss += ( relu(out_delta - gamma * delta).mean() * gamma2 )
    return loss


def spectral_regularization_lowpass(graph_encoder, x_src, edge_index_src, edge_attr_src, x_tgt, edge_index_tgt, edge_attr_tgt, gamma=1):
    node_num = x_src.shape[0]
    loss = 0

    try: eival, eivec = get_laplacian_evd(edge_index_src, node_num)
    except: return torch.zeros(1).cuda()
    x = torch.einsum('nm,md->nd', eivec, x_src)
    x_out = graph_encoder(x_src, edge_index_src, edge_attr_src)
    x_out = torch.einsum('nm,md->nd', eivec, x_out)
    loss += relu(x_out.abs() - gamma * x.abs()).mean()

    try: eival, eivec = get_laplacian_evd(edge_index_tgt, node_num)
    except: return torch.zeros(1).cuda()
    x = torch.einsum('nm,md->nd', eivec, x_tgt)
    x_out = graph_encoder(x_tgt, edge_index_tgt, edge_attr_tgt)
    x_out = torch.einsum('nm,md->nd', eivec, x_out)
    loss += relu(x_out.abs() - gamma * x.abs()).mean()
    return loss



'''
read_ppi('human')
read_ppi('yeast')
read_ppi('mouse')
read_ppi('fruit_fly')
read_ppi('zebrafish')
read_ppi('nematode')
'''

