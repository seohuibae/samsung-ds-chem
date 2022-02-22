import torch
import torch.nn as nn

import torch.nn.functional as F 
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MessagePassing

def create_batch_info(data, edge_counter):
    """ Compute some information about the batch that will be used by SMP."""
    x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.num_graphs

    # Compute some information about the batch
    # Count the number of nodes in each graph
    unique, n_per_graph = torch.unique(data.batch, return_counts=True)
    n_batch = torch.zeros_like(batch, dtype=torch.float)

    for value, n in zip(unique, n_per_graph):
        n_batch[batch == value] = n.float()

    # Count the average number of edges per graph
    dummy = x.new_ones((data.num_nodes, 1))
    average_edges = edge_counter(dummy, edge_index, batch, batch_size)

    # Aggregate into a dict
    batch_info = {'num_nodes': data.num_nodes,
                  'num_graphs': data.num_graphs,
                  'batch': data.batch,
                  'n_per_graph': n_per_graph,
                  'n_batch': n_batch[:, None, None].float(),
                  'average_edges': average_edges[:, :, None]}

    return batch_info

class EdgeCounter(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, batch, batch_size):
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return global_mean_pool(n_edges, batch, batch_size)[batch]


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, dropout=0.): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.dropout = dropout
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.FC_layers[self.L](y)
        return y

class MLPReadout2(nn.Module):

    def __init__(self, input_dim, output_dim, dims=[100], dropout=0.): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = []
        indim = input_dim
        for dim in dims: 
            list_FC_layers.append(nn.Linear(indim, dim, bias=True))
            indim=dim
        list_FC_layers.append(nn.Linear( indim , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = len(self.FC_layers)-1
        
        self.dropout = dropout
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.FC_layers[self.L](y)
        return y
