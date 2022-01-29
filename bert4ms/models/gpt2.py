import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Parameter
from .gpt import Conv1D, MLP


PRETRAINED_MODEL_ARCHIVE_MAP = {
}

PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

class Attention(nn.Cell):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx
        assert n_state % config.n_head == 0
        self.bias = Parameter(mnp.tril(mnp.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx), 'bias')
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(1-config.attn_pdrop)
        self.resid_dropout = nn.Dropout(1-config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = mnp.matmul(q, k)
        if self.scale:
            w = w / mnp.sqrt(v.shape[-1])
        nd, ns = w.shape[-2], w.shape[-1]
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax()(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask
        
        outputs = (mnp.matmul(w, v),)
        if self.output_attentions:
            outputs += (w,)
        return outputs

    def split_heads(self, x, k=False):
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(new_x_shape)
        if k:
            return x.transpose(0, 2, 3, 1)
        else:
            return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.transpose(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(new_x_shape)

    def construct(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = mnp.split(x, self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].swapaxes(-2, -1), layer_past[1]
            key = mnp.concatenate((past_key, key), axis=-1)
            value = mnp.concatenate((past_value, value), axis=-2)
        present = mnp.stack((key.swapaxes(-2, -1), value))

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = (a, present) + attn_outputs[1:]
        return outputs

class Block(nn.Cell):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embed
        self.ln_1 = nn.LayerNorm((nx,), epsilon=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm((nx,), eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def construct(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        a = output_attn[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = (x,) + output_attn[1:]
        return outputs

