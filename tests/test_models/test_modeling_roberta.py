import unittest
import pytest
import mindspore as ms
import numpy as np
from ddt import ddt, data
from cybertron.models import RobertaModel, RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from mindspore import Tensor

@ddt
class TestModelingRoberta(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RobertaConfig(vocab_size=1000,
                                    hidden_size=256,
                                    num_hidden_layers=4,
                                    num_attention_heads=4,
                                    intermediate_size=128)

    @data(True, False)
    def test_modeling_roberta(self, mode):
        model = RobertaModel(self.config)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            outputs, pooled = model(input_ids)
            return outputs, pooled
        
        if mode:
            forward = ms.jit(forward)

        outputs, pooled = forward(input_ids)
        assert outputs.shape == (1, 512, self.config.hidden_size)
        assert pooled.shape == (1, self.config.hidden_size)

    @pytest.mark.local
    def test_modeling_roberta_from_torch(self):
        model = RobertaModel.load('roberta-base', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    @data(True, False)
    def test_roberta_for_masked_lm(self, mode):
        model = RobertaForMaskedLM(self.config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            (prediction_scores,) = model(input_ids)
            return prediction_scores

        if mode:
            forward = ms.jit(forward)

        prediction_scores = forward(input_ids)
        assert prediction_scores.shape == (1, 512, model.config.vocab_size)

    @data(True, False)
    def test_roberta_for_sequence_classification(self, mode):
        model = RobertaForSequenceClassification(self.config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            (prediction_scores,) = model(input_ids)
            return prediction_scores

        if mode:
            forward = ms.jit(forward)
        prediction_scores = forward(input_ids)
        assert prediction_scores.shape == (1, model.config.num_labels)
