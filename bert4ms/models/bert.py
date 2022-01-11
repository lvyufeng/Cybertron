import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from ..common.activations import activation_map, GELU
from ..common.cell import PretrainedCell
from ..common.layers import Dense, Embedding
from ..common.tokenizers import FullTokenizer
from mindspore import ms_function, Tensor

class BertConfig:
    """Configuration for BERT
    """
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

class BertEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None):
        seq_len = input_ids.shape[1]
        position_ids = mnp.arange(seq_len)
        position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Cell):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(config.hidden_size, self.all_head_size)
        self.key = Dense(config.hidden_size, self.all_head_size)
        self.value = Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(Tensor(self.attention_head_size, mstype.float32))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer 

class BertSelfOutput(nn.Cell):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Cell):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def construct(self, input_tensor, attention_mask):
        self_output = self.self_attn(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Cell):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = activation_map.get(config.hidden_act, GELU())

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Cell):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = Dense(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Cell):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Cell):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = ()
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers += (hidden_states,)
        if not output_all_encoded_layers:
            all_encoder_layers += (hidden_states,)
        return all_encoder_layers

class BertPooler(nn.Cell):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size, activation='tanh')

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output

class BertModel(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

# class BertNextSentencePredict(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.classifier = Dense(config.hidden_size, 2)

#     def construct(self, h_pooled):
#         logits_clsf = self.classifier(h_pooled)
#         return logits_clsf

# class BertMaskedLanguageModel(nn.Cell):
#     def __init__(self, config, tok_embed_table):
#         super().__init__()
#         self.transform = Dense(config.hidden_size, config.hidden_size)
#         self.activation = activation_map.get(config.hidden_act, nn.GELU())
#         self.norm = nn.LayerNorm((config.hidden_size, ), epsilon=1e-12)
#         self.decoder = Dense(tok_embed_table.shape[1], tok_embed_table.shape[0], weight_init=tok_embed_table)

#     def construct(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.norm(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states

# class BertForPretraining(nn.Cell):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.nsp = BertNextSentencePredict(config)
#         self.mlm = BertMaskedLanguageModel(config, self.bert.embeddings.tok_embed.embedding_table)

#     def construct(self, input_ids, segment_ids):
#         outputs, h_pooled = self.bert(input_ids, segment_ids)
#         nsp_logits = self.nsp(h_pooled)
#         mlm_logits = self.mlm(outputs)
#         return mlm_logits, nsp_logits
