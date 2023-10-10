import torch
from torch import nn
from einops import rearrange
from model.utils import get_activation

#create position embedding sin/cos for each patch of size 100x7
def posemb_sincos_2d(x):
    # x is of shape (batch_size, seq_len, dim)
    seq_len = x.shape[1]
    dim = x.shape[2]
    pos = torch.arange(seq_len)[:, None]
    i = torch.arange(dim)[None, :]
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / dim)
    angle_rads = pos * angle_rates
    # apply sin to even index in the array
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_emb = angle_rads[None, :, :]

    return pos_emb

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, activation = 'relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, activation = 'relu'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim, activation)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x