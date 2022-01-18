import mindspore
import mindspore.numpy as mnp
from .bert import BertEmbeddings, BertModel

PRETRAINED_MODEL_ARCHIVE_MAP = {
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

