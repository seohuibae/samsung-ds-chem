import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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