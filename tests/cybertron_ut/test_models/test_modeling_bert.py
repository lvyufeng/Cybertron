import unittest
import pytest
import mindspore as ms
import numpy as np
from ddt import ddt, data
from cybertron.transformers import BertModel, BertConfig, BertForPretraining
from mindspore import Tensor

@ddt
class TestModelingBert(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BertConfig(vocab_size=1000,
                                 hidden_size=256,
                                 num_hidden_layers=4,
                                 num_attention_heads=4,
                                 intermediate_size=128)

    @data(True, False)
    def test_modeling_bert(self, mode):
        model = BertModel(self.config)
        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            outputs, pooled = model(input_ids)
            return outputs, pooled
        
        if mode:
            forward = ms.jit(forward)

        outputs, pooled = forward(input_ids)

        assert outputs.shape == (1, 512, self.config.hidden_size)
        assert pooled.shape == (1, self.config.hidden_size)

    @data(True, False)
    def test_modeling_bert_pretraining(self, mode):
        model = BertForPretraining(self.config)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)


        def forward(input_ids):
            mlm_logits, nsp_logits = model(input_ids)
            return mlm_logits, nsp_logits
        
        if mode:
            forward = ms.jit(forward)

        mlm_logits, nsp_logits = forward(input_ids)
        assert mlm_logits.shape == (1, 512, self.config.vocab_size)
        assert nsp_logits.shape == (1, 2)

    @pytest.mark.local
    def test_modeling_bert_from_torch(self):
        model = BertModel.load('bert-base-uncased', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)
