import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):        
    def __init__(self, gnn, mlp_head):
        super(Model, self).__init__()
        self.gnn = gnn 
        self.mlp_head = mlp_head 

    def forward(self, data, epoch=None):
        """ data.x: (num_nodes, num_features)"""
        x = self.gnn(data)
        # print(x.shape)
        # print(self.mlp_head)
        pred = self.mlp_head(x)
        return pred
