import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GATConv
from .model_utils import EdgeCounter, MLPReadout, create_batch_info

class GAT(nn.Module): 
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim, heads, dropout=0.6, readout='mean', mlp_dropout=0.5):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout 
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GATConv(num_input_features, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers-2): 
            self.layers.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_dim*heads, out_dim, heads=1, concat=False, dropout=dropout))

        self.mlp_head = MLPReadout(out_dim, 4, L=2, dropout=mlp_dropout)
    
    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x.float(), data.edge_index, data.batch, data.num_graphs
        for i in range(len(self.layers)-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            x = F.elu(x) # elu for gat 
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

class GATwoRegressor(nn.Module): 
    def __init__(self, num_input_features, num_layers, hidden_dim, out_dim, heads, dropout=0.6, readout='mean', mlp_dropout=0.5):
        super(GATwoRegressor, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout 
        self.readout = readout 

        self.layers = nn.ModuleList([])
        self.layers.append(GATConv(num_input_features, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers-2): 
            self.layers.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_dim*heads, out_dim, heads=1, concat=False, dropout=dropout))

        # self.mlp_head = MLPReadout(out_dim, 4, L=2, dropout=mlp_dropout)
        self.linears = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    
    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x.float(), data.edge_index, data.batch, data.num_graphs
        for i in range(len(self.layers)-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            x = F.elu(x) # elu for gat 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)

        if self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        # pred = self.mlp_head(x)
        pred = self.linears(x)
        # pred = x 

        return pred

    
    def __repr__(self):
        return self.__class__.__name__