import itertools
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from ..common.activations import GELU
from ..common.layers import Dense

class MultiHeadAttention(nn.Cell):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super(MultiHeadAttention, self).__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Dense(dim, dim)
        self.k_lin = Dense(dim, dim)
        self.v_lin = Dense(dim, dim)
        self.out_lin = Dense(dim, dim)

    def construct(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.shape
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.shape[1]
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.ndim == 3 else (bs, 1, 1, klen)

        q = self.q_lin(input).view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = self.k_lin(input).view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)                                      # (bs, n_heads, qlen, dim_per_head)
            v = self.v_lin(input).view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)                                      # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = self.k_lin(k).view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)                                          # (bs, n_heads, qlen, dim_per_head)
            v = self.v_lin(v).view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)                                          # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = mnp.concatenate([k_, k], axis=2)                     # (bs, n_heads, klen, dim_per_head)
                    v = mnp.concatenate([v_, v], axis=2)                     # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / ops.sqrt(ops.scalar_to_tensor(dim_per_head))                 # (bs, n_heads, qlen, dim_per_head)
        scores = ops.matmul(q, k.transpose(2, 3))                            # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)              # (bs, n_heads, qlen, klen)
        scores.masked_fill(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        weights = ops.Softmax()(scores.astype(mindspore.float32))            # (bs, n_heads, qlen, klen)
        weights = nn.Dropout(1 - self.dropout)(weights)                      # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = ops.matmul(weights, v)                                     # (bs, n_heads, qlen, dim_per_head)
        context = context.swapaxes(1, 2).view(bs, -1, self.n_heads * dim_per_head)                                            # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Cell):

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super(TransformerFFN, self).__init__()
        self.dropout = config.dropout
        self.lin1 = Dense(in_dim, dim_hidden)
        self.lin2 = Dense(dim_hidden, out_dim)
        self.act = GELU(False) if config.gelu_activation else nn.ReLU()
        self.dropout = nn.Dropout(1-self.dropout)

    def construct(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

