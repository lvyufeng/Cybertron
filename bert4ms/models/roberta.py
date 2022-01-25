import mindspore
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.common.initializer import initializer

from bert4ms.common.activations import GELU
from bert4ms.common.layers import Dense
from .bert import BertEmbeddings, BertModel, BertPretrainedCell
from ..configs.roberta import RobertaConfig

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://sharelist-lv.herokuapp.com/models/roberta/roberta-base.ckpt",
    "roberta-large": "https://sharelist-lv.herokuapp.com/models/roberta/roberta-large.ckpt",
    "roberta-large-mnli": "https://sharelist-lv.herokuapp.com/models/roberta/roberta-large-mnli.ckpt",
}

PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
]

class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 1
    
    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = mnp.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=mindspore.int64)
            position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        return super().construct(input_ids, token_type_ids, position_ids)

class RobertaModel(BertModel):
    pretrained_model_archive = PRETRAINED_MODEL_ARCHIVE_MAP
    pytorch_pretrained_model_archive_list = PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        return super().construct(input_ids, attention_mask, token_type_ids, position_ids, head_mask)

class RobertaForMaskedLM(BertPretrainedCell):
    pretrained_model_archive = PRETRAINED_MODEL_ARCHIVE_MAP
    pytorch_pretrained_model_archive_list = PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST
    config_class = RobertaConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.embedding_table

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = nn.SoftmaxCrossEntropyWithLogits(True, 'mean')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs   
    
class RobertaLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

        self.decoder = Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.bias = Parameter(initializer('zeros', config.vocab_size), 'bias')
        self.gelu = GELU()

    def construct(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x) + self.bias
        return x
