from ..common.activations import MultiHeadAttention, activation_map
from ..common.cell import Cell
from ..common.layers import Dense, Embedding
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp

class PoswiseFeedForwardNet(Cell):
    def __init__(self, d_model, d_ff, activation:str='gelu'):
        super().__init__()
        self.fc1 = Dense(d_model, d_ff)
        self.fc2 = Dense(d_ff, d_model)
        self.activation = activation_map.get(activation, nn.GELU())

    def construct(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class BertEmbeddings(Cell):
    def __init__(self, vocab_size, d_model, max_len, n_segments):
        super().__init__()
        self.tok_embed = Embedding(vocab_size, d_model)
        self.pos_embed = Embedding(max_len, d_model)
        self.seg_embed = Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm((d_model,))

        self.expand_dims = P.ExpandDims()

    def construct(self, x, seg):
        seq_len = x.shape[1]
        pos = mnp.arange(seq_len)
        pos = P.BroadcastTo(x.shape)(self.expand_dims(pos, 0))
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class BertEncoderLayer(Cell):
    def __init__(self, d_model, n_heads, d_ff, activation, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, activation)

    def construct(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BertEncoder(Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList([BertEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.activation) for _ in range(config.n_layers)])

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class BertModel(Cell):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config.vocab_size, config.d_model, config.max_len, config.n_segments)
        self.encoder = BertEncoder(config)
        self.pooler = Dense(config.d_model, config.d_model, activation='tanh')
        
    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

    @staticmethod
    def load(self):
        pass

class BertForPretraining(Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)