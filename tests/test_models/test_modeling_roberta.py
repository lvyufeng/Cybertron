import unittest
import mindspore
import torch
import numpy as np
from bert4ms.models import RobertaModel, RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from mindspore import Tensor
from mindspore import context
from transformers import RobertaModel as ptRobertaModel

class TestModelingRoberta(unittest.TestCase):
    def test_modeling_roberta_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = RobertaConfig()
        model = RobertaModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_roberta_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = RobertaConfig()
        model = RobertaModel(config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_roberta_from_torch(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = RobertaModel.load('roberta-base', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_roberta_with_ckpt_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = RobertaModel.load('roberta-base')
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        outputs, pooled = model(ms_input_ids)
        
        pt_model = ptRobertaModel.from_pretrained('roberta-base')
        pt_model.eval()
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        outputs_pt, pooled_pt = pt_model(input_ids=pt_input_ids)

        assert np.allclose(outputs.asnumpy(), outputs_pt.detach().numpy(), atol=1e-3)
        assert np.allclose(pooled.asnumpy(), pooled_pt.detach().numpy(), atol=1e-3)
    
    def test_roberta_for_masked_lm(self):
        context.set_context(mode=context.GRAPH_MODE)
        model = RobertaForMaskedLM.load('roberta-base')
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (prediction_scores,) = model(input_ids)
        assert prediction_scores.shape == (1, 512, model.config.vocab_size)

    def test_roberta_for_sequence_classification(self):
        context.set_context(mode=context.GRAPH_MODE)
        model = RobertaForSequenceClassification.load('roberta-large-mnli')
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (prediction_scores,) = model(input_ids)
        assert prediction_scores.shape == (1, model.config._num_labels)