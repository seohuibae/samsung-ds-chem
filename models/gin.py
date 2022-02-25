import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GINConv
from .model_utils import EdgeCounter, MLPReadout, create_batch_info

class GIN(torch.nn.Module):        
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim, dropout=0.1, readout='sum', mlp_dropout=0.5): # sum by default
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout 
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GINConv(nn.Sequential(nn.Linear(num_input_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        for _ in range(num_layers-2): 
            self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, out_dim), nn.ReLU())))

        self.mlp_head = MLPReadout(out_dim, 4, L=2, dropout=mlp_dropout)
    
    def forward(self, data):
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
    
    def __repr__(self):
        return self.__class__.__name__

class GINwoRegressor(torch.nn.Module):        
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim, dropout=0.1, readout='sum', mlp_dropout=0.5): # sum by default
        super(GINwoRegressor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout 
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GINConv(nn.Sequential(nn.Linear(num_input_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        for _ in range(num_layers-2): 
            self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, out_dim), nn.ReLU())))

        # self.mlp_head = MLPReadout(out_dim, 4, L=2, dropout=mlp_dropout)
    
    def forward(self, data):
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
    
    def __repr__(self):
        return self.__class__.__name__