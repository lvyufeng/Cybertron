from typing import Optional
from .cell import Cell
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, XavierUniform, Zero
from .layers import Dense

ActMap = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}

class MultiheadAttention(Cell):
    def __init__(self, embed_dim, num_heads, dropout=0., has_bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim)), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim)), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim)), 'v_proj_weight')
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim)), 'in_proj_weight')

        if has_bias:
            self.in_proj_bias = Parameter(initializer(Zero(), (3 * embed_dim)), 'in_proj_bias')

        self.out_proj = Dense(embed_dim, embed_dim,has_bias=has_bias)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierUniform(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = Parameter(initializer(XavierUniform(), (1, 1, embed_dim)), 'bias_v')

        self.add_zero_attn = add_zero_attn

        self.transpose = P.Transpose()
    def construct(self, query:Tensor, key:Tensor, value:Tensor, key_padding_mask: Optional[Tensor]=None):
        if self.batch_first:
            query = self.transpose(query, (1, 0, 2))
            key = self.transpose(key, (1, 0, 2))
            value = self.transpose(value, (1, 0, 2))
        return None