from torch import nn
from torch.nn import functional as F

def get_activation(name):
    if name == 'relu':
        act = nn.ReLU()
    elif name == 'tanh':
        act = nn.Tanh()
    elif name == 'sigmoid':
        act = nn.Sigmoid()
    elif name == 'gelu':
        act = nn.GELU()
    elif name == 'prelu':
        act = nn.PReLU()
    elif name == 'elu':
        act = nn.ELU()
    elif name == 'lrelu':
        act = nn.LeakyReLU()
    elif name == 'identity':
        act = nn.Identity()
    return act

class ReLUX(nn.Module):
    def __init__(self, max_value: float=1.0):
        super(ReLUX, self).__init__()
        self.max_value = float(max_value)
        self.scale     = 6.0/self.max_value

    def forward(self, x):
        return F.relu6(x * self.scale) / (self.scale)