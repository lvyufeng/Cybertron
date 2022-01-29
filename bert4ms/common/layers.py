import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
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

class SequenceSummary(nn.Cell):
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = ops.Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = Dense(config.hidden_size, num_classes)

        self.activation = ops.Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()

        self.first_dropout = ops.Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = ops.Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(1-config.summary_last_dropout)

    def construct(self, hidden_states, cls_index=None):
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(axis=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = mnp.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=mindspore.int32)
            else:
                cls_index = cls_index.expand_dims(-1).expand_dims(-1)
                cls_index = mnp.broadcast_to(cls_index, (-1,) * (cls_index.ndim-1) + (hidden_states.shape[-1],))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = ops.gather_d(hidden_states, -2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output