import math
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform, _calculate_fan_in_and_fan_out
from mindspore import Tensor

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None):
        if weight_init is None:
            weight_init = initializer(HeUniform(math.sqrt(5)), (out_channels, in_channels))
        if bias_init is None:
            fan_in, _ = _calculate_fan_in_and_fan_out((out_channels, in_channels))
            bound = 1 / math.sqrt(fan_in)
            bias_init = initializer(Uniform(bound), (out_channels))
        super().__init__(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init, has_bias=has_bias, activation=activation)

class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32, padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings:Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding