import pdb
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GINEConv
from sinkhorn_transformer import SinkhornTransformerLM


class transformer(nn.Module):
    def __init__(self, hidden_dim=256, bucket_size=64):
        super().__init__()
        self.transformer = SinkhornTransformerLM(num_tokens=29, dim=hidden_dim, depth=4, heads=4, bucket_size=bucket_size, max_seq_len=1536, reversible=True, return_embeddings=True).cuda()

    def forward(self, seq_list):
        def tokenization(seq):
            if len(seq) > 1536:
                seq = seq[:1536]
            token = list( map(lambda c: {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}[c], seq) )
            x = torch.zeros(1536, dtype=torch.int64)
            x[:len(token)] = torch.tensor(token)
            return x.reshape((1, 1536))

        token = list(map(tokenization, seq_list))
        token = torch.cat(token, dim=0).to('cuda')
        x_mask = (token>0).to('cuda')
        x = self.transformer(token, input_mask=x_mask)

        x_mask_int = torch.zeros(x.shape).to('cuda')
        x_mask_int[token>0] = 1
        x = (x * x_mask_int).sum(dim=1) / x_mask_int.sum(dim=1)
        return x

    def forward_nopool(self, seq_list):
        def tokenization(seq):
            if len(seq) > 1536:
                seq = seq[:1536]
            token = list( map(lambda c: {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}[c], seq) )
            x = torch.zeros(1536, dtype=torch.int64)
            x[:len(token)] = torch.tensor(token)
            return x.reshape((1, 1536))

        token = list(map(tokenization, seq_list))
        token = torch.cat(token, dim=0).to('cuda')
        x_mask = (token>0).to('cuda')
        x = self.transformer(token, input_mask=x_mask)
        x = x[x_mask]
        return x


class hrnn(nn.Module):
    def __init__(self, hidden_dim=256, kmer=50):
        super().__init__()
        self.aa_embedding = nn.Embedding(29, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.kmer = kmer

    def forward(self, seq_list):
        def tokenization(seq):
            if len(seq) > 1500:
                seq = seq[:1500]
            token = list( map(lambda c: {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}[c], seq) )
            x = torch.zeros(1500, dtype=torch.int64)
            x[:len(token)] = torch.tensor(token)
            return x.reshape((1, 1500))

        token_list = list(map(tokenization, seq_list))
        token = torch.cat(token_list, dim=0).to('cuda')

        x = self.aa_embedding(token)
        b, l, d = x.shape
        x = x.reshape((b, l//self.kmer, self.kmer, d))
        b, l, k, d = x.shape

        x = x.reshape((b*k, l, d))
        x, _ = self.gru1(x)
        x = x.reshape((b*l, k, d))
        x, _ = self.gru2(x)
        x = x.reshape ((b, l*k, d))

        x_mask = torch.zeros(x.shape).to('cuda')
        x_mask[token>0] = 1
        x = (x * x_mask).sum(dim=1) / x_mask.sum(dim=1)
        return x


class gat(nn.Module):
    def __init__(self, layer_num=3, hidden_dim=256, attn_head=4):
        super().__init__()
        # self.gatconv_list = nn.ModuleList( [GATConv(in_channels=1024, out_channels=hidden_dim//attn_head, heads=attn_head, edge_dim=3)] +
        #                                    [GATConv(in_channels=hidden_dim, out_channels=hidden_dim//attn_head, heads=attn_head, edge_dim=3) for _ in range(layer_num-1)] )
        self.gatconv_list = nn.ModuleList( [GATConv(in_channels=hidden_dim, out_channels=hidden_dim//attn_head, heads=attn_head, edge_dim=3) for _ in range(layer_num)] )
        self.relu = nn.ReLU()
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 self.relu,
                                 nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for gatconv in self.gatconv_list:
            x0 = x ###
            x = gatconv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.relu(x)
            x = x + x0 ###
        x = self.proj(x)
        return x


class mlp(nn.Module):
    def __init__(self, layer_num=3, hidden_dim=256):
        super().__init__()
        self.in_channels, self.out_channels = hidden_dim, hidden_dim
        self.relu = nn.ReLU()
        self.linear_list = nn.ModuleList( [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num)] )
    def forward(self, x):
        for n, linear in enumerate(self.linear_list):
            x = linear(x)
            if n == len(self.linear_list) - 1:
                break
            x = self.relu(x)
        return x


class gin(nn.Module):
    def __init__(self, layer_num=3, hidden_dim=256, mlp_layer=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.ginconv_list = nn.ModuleList( [GINEConv(nn.Sequential(mlp(mlp_layer, hidden_dim)), edge_dim=3) for _ in range(layer_num)] )
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                  self.relu,
                                  nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for ginconv in self.ginconv_list:
            x0 = x ###
            x = ginconv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.relu(x)
            x = x + x0 ###
        x = self.proj(x)
        return x

