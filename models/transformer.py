import torch 
import torch.nn as nn 

import math 

from .model_utils import  MLPReadout

class SimpleTextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(SimpleTextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.mlp_head = MLPReadout(embed_dim, 4, L=2, dropout=0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for i in range(3):
            self.mlp_head.FC_layers[i].weight.data.uniform_(-initrange, initrange)
            self.mlp_head.FC_layers[i].bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output = self.mlp_head(embedded)
        return output

class Transformer(nn.Module):
 
    def __init__(self, vocab_size, embed_dim, num_class, num_layers, nhead, d_model, dropout=0.1):
        super(Transformer, self).__init__()
        assert embed_dim == d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim 
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True) for _ in range(num_layers)])
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nhead, d_model, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.mlp_head = MLPReadout(d_model, 4, L=2, dropout=0.5)

        self.src_mask = None 
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for i in range(3):
            self.mlp_head.FC_layers[i].weight.data.uniform_(-initrange, initrange)
            self.mlp_head.FC_layers[i].bias.data.zero_()

    def forward(self, src, offsets):
        emb = self.embedding(src, offsets) * math.sqrt(self.embed_dim)
        seq_len = emb.size(0) # or src.size(1)
        if self.src_mask is None or self.src_mask.size(0) != seq_len: 
            device = emb.device 
            mask = self._generate_square_subsequent_mask(seq_len).to(device)
            self.src_mask = mask

        output = self.pos_encoder(emb)
        output = self.encoder(output, self.src_mask)
        output = self.mlp_head(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)