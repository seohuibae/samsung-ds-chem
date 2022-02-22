import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv
from .model_utils import EdgeCounter, MLPReadout, MLPReadout2, create_batch_info

class GCN(torch.nn.Module):        
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim, dropout=0.1, readout='mean', mlp_dropout=0.5, mlp_dims=[100]):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GCNConv(num_input_features, hidden_dim))
        for _ in range(num_layers-2): 
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, out_dim))

        self.mlp_head = MLPReadout2(out_dim, 4, dims=mlp_dims, dropout=mlp_dropout)
    
    def forward(self, data, epoch=None):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x.float(), data.edge_index, data.batch, data.num_graphs

        for i in range(len(self.layers)-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)

        if self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        pred = self.mlp_head(x)

        return pred
    
    def loss_cl(self, x1, x2):
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

    def __repr__(self):
        return self.__class__.__name__
    

class GCNwoFC(torch.nn.Module):        
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim=4, dropout=0.1, readout='mean', mlp_dropout=None):
        super(GCNwoFC, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GCNConv(num_input_features, hidden_dim))
        for _ in range(num_layers-2): 
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, out_dim))

        # self.mlp_head = MLPReadout(out_dim, 4, L=2, dropout=mlp_dropout)
    
    def forward(self, data, epoch=None):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x.float(), data.edge_index, data.batch, data.num_graphs

        for i in range(len(self.layers)-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)

        if self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        # pred = self.mlp_head(x)
        pred = x 

        return pred