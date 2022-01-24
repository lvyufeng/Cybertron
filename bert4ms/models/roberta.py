import mindspore
import mindspore.numpy as mnp
from .bert import BertEmbeddings, BertModel
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