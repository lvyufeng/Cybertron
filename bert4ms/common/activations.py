from typing import Optional
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, XavierUniform, Zero
from .cell import Cell
from .layers import Dense
from .utils import MaskedFill

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
}

class ScaledDotProductAttention(Cell):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.scale = Tensor(d_k, mindspore.float32)
        self.matmul = nn.MatMul()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax(axis=-1)
        self.sqrt = P.Sqrt()
        self.masked_fill = MaskedFill(-1e9)

        if dropout > 0.0:
            self.dropout = nn.Dropout(1-dropout)
        else:
            self.dropout = None

    def construct(self, Q, K, V, attn_mask):
        K = self.transpose(K, (0, 1, 3, 2))
        scores = self.matmul(Q, K) / self.sqrt(self.scale) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = self.masked_fill(scores, attn_mask) # Fills elements of self tensor with value where mask is one.
        attn = self.softmax(scores)
        context = self.matmul(attn, V)
        if self.dropout is not None:
            context = self.dropout(context)
        return context, attn

class MultiHeadAttention(Cell):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.W_Q = Dense(d_model, d_model)
        self.W_K = Dense(d_model, d_model)
        self.W_V = Dense(d_model, d_model)
        self.linear = Dense(d_model, d_model)
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "embed_dim must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm((d_model, ), epsilon=1e-5)
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        # ops
        self.transpose = P.Transpose()
        self.expanddims = P.ExpandDims()
        self.tile = P.Tile()
        
    def construct(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_Q(Q).view((batch_size, -1, self.n_heads, self.head_dim)) 
        k_s = self.W_K(K).view((batch_size, -1, self.n_heads, self.head_dim)) 
        v_s = self.W_V(V).view((batch_size, -1, self.n_heads, self.head_dim)) 
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.transpose(q_s, (0, 2, 1, 3)) # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.transpose(k_s, (0, 2, 1, 3)) # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.transpose(v_s, (0, 2, 1, 3)) # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = self.expanddims(attn_mask, 1)
        attn_mask = self.tile(attn_mask, (1, self.n_heads, 1, 1)) # attn_mask : [batch_size x n_heads x len_q x len_k]
        
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = self.transpose(context, (0, 2, 1, 3)).view((batch_size, -1, self.n_heads * self.head_dim)) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context) 
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]