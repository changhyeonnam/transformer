import torch
import torch.nn as nn
from model.Multi_Head import MultiHeadAttentionLayer
from model.Pos_Encoding import PositionwiseFFLayer

class Encoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hid_dim:int,
                 n_layers:int,
                 n_heads,
                 pf_dim:int,
                 dropout:float,
                 max_length=100):
        super(Encoder,self).__init__()

        # token embedding
        self.tok_embedding = nn.Embedding(input_dim,hid_dim)

        # positional embedding for sequence order
        # max_length : accept sentences up to 100 tokens long.
        # original paper use fixed static embedding but i used embedding layer for positional embedding.
        self.pos_embedding = nn.Embedding(max_length,hid_dim)

        # scale vector
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

        self.layers = nn.ModuleList([EncoderLayer(hid_dim=hid_dim,
                                                  n_heads=n_heads,
                                                  pff_dim=pf_dim,
                                                  dropout=dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create index (0,src_len-1) for each batch.
        # shape of pos is (batch_size, src_len)
        pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1)

        # token_embedding and positional embedding are elementwise summed.
        # before summation, token embeddings are multipled by scaling factor sqrt(d_model=hid_dim)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src pass through N encoder layers.
        # src_mask is same length of source mask. if token is <pad> token, it's mask value is 0 otherwise 1.
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim:int,
                 n_heads:int,
                 pff_dim:int,
                 dropout:float):

        super(EncoderLayer,self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        # to attend soruce sentence itself. we call it self-attention.
        self.MultiHead_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFFLayer(hid_dim, pff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #1 pass soruce sentence, souce sentence mask in to multi-head attention layer
        _src, _ =self.MultiHead_attention(src,src,src,src_mask)
        #2 apply residual connection in layer norm. pass _src thorugh this combined layer.
        src = self.self_attention_layer_norm(src + self.dropout(_src))
        #3 pass src thorugh position wise ff layer
        _src = self.positionwise_feedforward(src)
        #4 again apply layer norm in ff_layer
        src = self.ff_layer_norm(src+ self.dropout(_src))
        return src