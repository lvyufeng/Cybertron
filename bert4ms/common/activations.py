import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from .layers import Dense

class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.scale = Tensor(d_k, mindspore.float32)
        self.softmax = nn.Softmax(axis=-1)

        if dropout > 0.0:
            self.dropout = nn.Dropout(1-dropout)
        else:
            self.dropout = None

    def construct(self, query, key, value, attn_mask):
        key = key.transpose((0, 1, 3, 2))
        scores = ops.matmul(query, key) / ops.sqrt(self.scale) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores = self.masked_fill(scores, attn_mask) # Fills elements of self tensor with value where mask is one.
        scores = scores.masked_fill(attn_mask, -1e9)
        # scores = scores + attn_mask
        attn = self.softmax(scores)
        context = ops.matmul(attn, value)
        if self.dropout is not None:
            context = self.dropout(context)
        return context, attn

class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.query = Dense(d_model, d_model)
        self.key = Dense(d_model, d_model)
        self.value = Dense(d_model, d_model)
        self.linear = Dense(d_model, d_model)
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "embed_dim must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm((d_model, ), epsilon=1e-12)
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        
    def construct(self, query, key, value, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = query, query.shape[0]
        q_s = self.query(query).view((batch_size, -1, self.n_heads, self.head_dim)) 
        k_s = self.key(key).view((batch_size, -1, self.n_heads, self.head_dim)) 
        v_s = self.value(value).view((batch_size, -1, self.n_heads, self.head_dim)) 
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = q_s.transpose((0, 2, 1, 3)) # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = k_s.transpose((0, 2, 1, 3)) # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = v_s.transpose((0, 2, 1, 3)) # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.expand_dims(1)
        attn_mask = ops.tile(attn_mask, (1, self.n_heads, 1, 1)) # attn_mask : [batch_size x n_heads x len_q x len_k]
        
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose((0, 2, 1, 3)).view((batch_size, -1, self.n_heads * self.head_dim)) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context) 
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(approximate=False),
}