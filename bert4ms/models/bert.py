import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from ..common.activations import MultiHeadAttention, activation_map
from ..common.cell import PretrainedCell
from ..common.layers import Dense, Embedding
from ..common.tokenizers import FullTokenizer
from mindspore import ms_function

@ms_function
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    pad_attn_mask = ops.equal(seq_k, 0)
    pad_attn_mask = pad_attn_mask.expand_dims(1)
    # pad_attn_mask = P.ExpandDims()(P.Equal()(seq_k, 0), 1)
    # pad_attn_mask = P.Cast()(pad_attn_mask, mstype.int32)
    pad_attn_mask = ops.BroadcastTo((batch_size, len_q, len_k))(pad_attn_mask)
    # pad_attn_mask = P.Cast()(pad_attn_mask, mstype.bool_)
    return pad_attn_mask

class BertTokenizer(FullTokenizer):
    def __init__(self, vocab_file, do_lower_case):
        super().__init__(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, ):
        return None

class BertConfig:
    def __init__(self,
                seq_length=128,
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self, d_model, d_ff, activation:str='gelu'):
        super().__init__()
        self.fc1 = Dense(d_model, d_ff)
        self.fc2 = Dense(d_ff, d_model)
        self.activation = activation_map.get(activation, nn.GELU(False))
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-12)

    def construct(self, inputs):
        residual = inputs
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        
        outputs = self.fc2(outputs)
        return self.layer_norm(outputs + residual)

class BertEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = Embedding(config.max_position_embeddings, config.hidden_size)
        self.seg_embed = Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)

    def construct(self, x, seg):
        seq_len = x.shape[1]
        pos = mnp.arange(seq_len)
        pos = pos.expand_dims(0).expand_as(x)
        seg_embedding = self.seg_embed(seg)
        tok_embedding = self.tok_embed(x)
        embedding = tok_embedding + self.pos_embed(pos) + seg_embedding
        # embedding = self.tok_embed(x) + self.seg_embed(seg)
        return self.norm(embedding)

class BertEncoderLayer(nn.Cell):
    def __init__(self, d_model, n_heads, d_ff, activation, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, activation)

    def construct(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList([BertEncoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_act, config.hidden_dropout_prob) for _ in range(config.num_hidden_layers)])

    def construct(self, inputs, enc_self_attn_mask):
        outputs = inputs
        for layer in self.layers:
            outputs, enc_self_attn = layer(outputs, enc_self_attn_mask)
            # print(outputs)
        return outputs

class BertModel(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = Dense(config.hidden_size, config.hidden_size, activation='tanh')
        
    def construct(self, input_ids, segment_ids):
        outputs = self.embeddings(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # print(enc_self_attn_mask)
        outputs = self.encoder(outputs, enc_self_attn_mask)
        h_pooled = self.pooler(outputs[:, 0]) 
        return outputs, h_pooled

class BertNextSentencePredict(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.classifier = Dense(config.hidden_size, 2)

    def construct(self, h_pooled):
        logits_clsf = self.classifier(h_pooled)
        return logits_clsf

class BertMaskedLanguageModel(nn.Cell):
    def __init__(self, config, tok_embed_table):
        super().__init__()
        self.transform = Dense(config.hidden_size, config.hidden_size)
        self.activation = activation_map.get(config.hidden_act, nn.GELU())
        self.norm = nn.LayerNorm((config.hidden_size, ), epsilon=1e-12)
        self.decoder = Dense(tok_embed_table.shape[1], tok_embed_table.shape[0], weight_init=tok_embed_table)

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertForPretraining(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.nsp = BertNextSentencePredict(config)
        self.mlm = BertMaskedLanguageModel(config, self.bert.embeddings.tok_embed.embedding_table)

    def construct(self, input_ids, segment_ids):
        outputs, h_pooled = self.bert(input_ids, segment_ids)
        nsp_logits = self.nsp(h_pooled)
        mlm_logits = self.mlm(outputs)
        return mlm_logits, nsp_logits
