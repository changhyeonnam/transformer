import torch
import torch.nn as nn
from model.Multi_Head import MultiHeadAttentionLayer
from model.Pos_Encoding import PositionwiseFFLayer
class Decoder(nn.Module):
    '''
    Decoder model has two multi-head attention layers.
    1. self attention
    2. encoder attention
    '''
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=100):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim,output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super(DecoderLayer,self).__init__()
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFFLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, encoder_src, trg_mask, src_mask):
        # trg_mask is used to block future sequence.
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # layer norm + residual connection
        trg = self.self_attention_layer_norm(trg+self.dropout(_trg))

         # src mask is for preventing multi-head attention form attedning to <pad> token with soruce sentence.
        _trg, attention = self.encoder_attention(trg, encoder_src, encoder_src, src_mask)
        # layer norm + residual connection
        trg = self.encoder_attention_layer_norm(trg+ self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention