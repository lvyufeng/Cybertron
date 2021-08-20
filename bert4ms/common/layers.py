import math
from mindspore.common import initializer
from .cell import Cell
from .activations import MultiheadAttention
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Zero, HeUniform

class TransformerEncoderLayer(Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class TransformerDecoderLayer(Cell):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bacht_first=batch_first)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class TransformerEncoder(Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class TransformerDecoder(Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            self.bias.set_data(initializer(Zero(), [self.out_channels]))