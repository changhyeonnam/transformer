import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # hid_dim split in to h heads so,  id_dim(d_model) is divided by number of heads
        self.head_dim = hid_dim// n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self,query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q,K,V's shape is (batch_size, head_dim, -1, n_heads)
        # split int to n_heads using view
        Q = Q.view(batch_size,-1,self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size,-1,self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size,-1,self.n_heads, self.head_dim).permute(0,2,1,3)

        # reshaped Key's shape (batch_size, head_dim, n_heads, -1)
        # dot product between query and key and scale by d_k (self.scale = head dimension)
        dot_product = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale

        # masking
        if mask is not None:
            dot_product = dot_product.masked_fill(mask==0,-1e10)

        # apply softmax and dropout
        attention = torch.softmax(dot_product, dim=-1)

        # apply attention to the value
        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x= self.fc_o(x)

        return x, attention