import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils import  MultipleOptimizer

def scaled_l2(pred, y, scale):
    return ((pred - y) ** 2 / scale).mean()

def mse(pred, y):
    criterion = nn.MSELoss()
    return criterion(pred, y)

def loss_cl(x1, x2):
    T = 0.1 
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

def build_optimizer(model, optim, lr, weight_decay, reg='all'): # reg: all, fc, gnn
    if optim=='adam':
        if reg == 'fc':
            optimizer = torch.optim.Adam([
                {'params':model.base_params()},
                {'params':model.classifier_params(), 'weight_decay': weight_decay}], lr=lr)
        elif reg == 'gnn':
            optimizer = torch.optim.Adam([
                {'params':model.base_params(), 'weight_decay': weight_decay},
                {'params':model.classifier_params()}], lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif optim == 'sparseadam':
        dense = []
        sparse = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # TODO: Find a better way to check for sparse gradients.
            if 'embed' in name:
                sparse.append(param)
            else:
                dense.append(param)
        optimizer = MultipleOptimizer(
            [torch.optim.Adam(
                dense,
                lr=lr, weight_decay=weight_decay),
            torch.optim.SparseAdam(
                sparse,
                lr=lr, weight_decay=weight_decay)])
    else:
        raise NotImplementedError
    return optimizer