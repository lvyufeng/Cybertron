import unittest
import pytest
import mindspore as ms
import numpy as np
from ddt import ddt, data
from cybertron.models import BartConfig, BartModel
from mindspore import Tensor

@ddt
class TestModelingBert(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BartConfig(vocab_size=1000,
                                 d_model=256,
                                 encoder_ffn_dim=1024,
                                 encoder_layers=4,
                                 encoder_attention_heads=8,
                                 decoder_ffn_dim=1024,
                                 decoder_layers=4,
                                 decoder_attention_heads=8)

    @data(True, False)
    def test_modeling_bert(self, mode):
        model = BartModel(self.config)
        input_ids = model.dummy_inputs['input_ids']

        def forward(input_ids):
            decoder_outputs, encoder_outputs = model(input_ids)
            return decoder_outputs, encoder_outputs
        
        if mode:
            forward = ms.jit(forward)

        decoder_outputs, encoder_outputs = forward(input_ids)

        assert decoder_outputs.shape == (*input_ids.shape, self.config.d_model)
        assert encoder_outputs.shape == (*input_ids.shape, self.config.d_model)

    # @data(True, False)
    # def test_modeling_bert_pretraining(self, mode):
    #     model = BertForPretraining(self.config)

    #     input_ids = Tensor(np.random.randn(1, 512), ms.int32)


    #     def forward(input_ids):
    #         mlm_logits, nsp_logits = model(input_ids)
    #         return mlm_logits, nsp_logits
        
    #     if mode:
    #         forward = ms.jit(forward)

    #     mlm_logits, nsp_logits = forward(input_ids)
    #     assert mlm_logits.shape == (1, 512, self.config.vocab_size)
    #     assert nsp_logits.shape == (1, 2)

    # @pytest.mark.local
    # def test_modeling_bert_from_torch(self):
    #     model = BertModel.load('bert-base-uncased', from_torch=True)

    #     input_ids = Tensor(np.random.randn(1, 512), ms.int32)

    #     outputs, pooled = model(input_ids)
    #     assert outputs.shape == (1, 512, 768)
    #     assert pooled.shape == (1, 768)
