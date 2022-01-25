import torch
import torch.nn as nn
from Multi_Head import MultiHeadAttentionLayer
from Pos_Encoding import PositionwiseFFLayer

class Encoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hid_dim:int,
                 n_layers:int,
                 pf_dim:int,
                 dropout:float,
                 max_length:int=100):
        super(Encoder,self).__init__()

        self.tok_embedding = nn.Embedding(input_dim,hid_dim)
        self.pos_embedding = nn.Embedding(max_length,hid_dim)

        self.layers = nn.Modulelist([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout) for _ in range(n_layers)])
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim:int,
                 n_heads:int,
                 pf_dim:int,
                 dropout:float,
                 device):
        super(EncoderLayer,self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFFLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        _src, _ =self.self_attention(src,src,src,src_mask)
        src = self.self_attention_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src+ self.dropout(_src))
        return src