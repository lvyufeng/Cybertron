import os
import logging
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.initializer import initializer
from ..common.activations import activation_map, GELU
from ..common.cell import PretrainedCell
from ..common.layers import Dense, Embedding
from ..configs.bert import BertConfig

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://sharelist-lv.herokuapp.com/models/bert/bert-base-uncased.ckpt",
    "bert-large-uncased": "https://sharelist-lv.herokuapp.com/models/bert/bert-large-uncased.ckpt",
    "bert-base-cased": "https://sharelist-lv.herokuapp.com/models/bert/bert-base-cased.ckpt",
    "bert-large-cased": "https://sharelist-lv.herokuapp.com/models/bert/bert-large-cased.ckpt",
    "bert-base-multilingual-uncased": "https://sharelist-lv.herokuapp.com/models/bert/bert-base-multilingual-uncased.ckpt",
    "bert-base-multilingual-cased": "https://sharelist-lv.herokuapp.com/models/bert/bert-base-multilingual-cased.ckpt",
    "bert-base-chinese": "https://sharelist-lv.herokuapp.com/models/bert/bert-base-chinese.ckpt"
}

PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
]

def torch_to_mindspore(pth_file):
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")

    from mindspore import Tensor
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embeddings' in k:
            k = k.replace('weight', 'embedding_table')
        if 'self' in k:
            k = k.replace('self', 'self_attn')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('.bin','.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')

    return ms_ckpt_path

class BertEmbeddings(nn.Cell):
    """Embeddings for BERT, include word, position and token_type
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        seq_len = input_ids.shape[1]
        if position_ids is None:
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
    """Self attention layer for BERT
    """
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        
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

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(ops.scalar_to_tensor(self.attention_head_size, mstype.float32))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

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

    def construct(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self_attn(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BertIntermediate(nn.Cell):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = activation_map.get(config.hidden_act, GELU(False))

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

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class BertEncoder(nn.Cell):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)
        return outputs

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

class BertPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size)
        self.transform_act_fn = activation_map.get(config.hidden_act, GELU(False))
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(initializer('zeros', config.vocab_size), 'bias')

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Cell):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Cell):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertPretrainedCell(PretrainedCell):
    pretrained_model_archive = PRETRAINED_MODEL_ARCHIVE_MAP
    pytorch_pretrained_model_archive_list = PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST
    config_class = BertConfig
    convert_torch_to_mindspore = torch_to_mindspore

class BertModel(BertPretrainedCell):
    """"""
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.num_hidden_layers = self.config.num_hidden_layers

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
                head_mask = mnp.broadcast_to(head_mask, (self.num_hidden_layers, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertForPretraining(BertPretrainedCell):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.embedding_table
    
        self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(True, 'mean')

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = self.loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs
        return outputs

class BertForMaskedLM(BertPretrainedCell):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.embedding_table

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[:2]
        if masked_lm_labels is not None:
            loss_fct = nn.SoftmaxCrossEntropyWithLogits(True, 'mean')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs