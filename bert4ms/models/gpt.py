import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal
from ..common.activations import activation_map, GELU

class Conv1D(nn.Cell):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = Parameter(initializer(Normal(0.02), (nx, nf)), 'weight')
        self.bias = Parameter(initializer('zeros', nf), 'bias')
    
    def construct(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = x.view(-1, x.shape[-1]) @ self.weight + self.bias
        x = x.view(size_out)
        return x

class MLP(nn.Cell):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embed
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = activation_map.get('gelu_approximate', GELU())

    def construct(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2