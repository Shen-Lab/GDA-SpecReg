import networkx as nx
import numpy as np
import torch
import torch_geometric as tgeom
import random
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import pdb


def calculate_gradient_penalty(critic, x_src, x_tgt):
    x = torch.cat([x_src, x_tgt], dim=0).requires_grad_(True)
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

