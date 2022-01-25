import unittest
import mindspore
import torch
import numpy as np
from bert4ms.models import BertModel, BertConfig, BertForPretraining
from mindspore import Tensor
from mindspore import context
from transformers import BertModel as ptBertModel

class TestModelingBert(unittest.TestCase):
    def test_modeling_bert_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = BertConfig()
        model = BertModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = BertConfig()
        model = BertModel(config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_pretraining_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = BertConfig()
        model = BertForPretraining(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        mlm_logits, nsp_logits = model(input_ids)
        assert mlm_logits.shape == (1, 512, config.vocab_size)
        assert nsp_logits.shape == (1, 2)

    def test_modeling_bert_pretraining_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = BertConfig()
        model = BertForPretraining(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        mlm_logits, nsp_logits = model(input_ids)
        assert mlm_logits.shape == (1, 512, config.vocab_size)
        assert nsp_logits.shape == (1, 2)

    def test_modeling_bert_from_torch(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = BertModel.load('bert-base-uncased', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_with_ckpt_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = BertModel.load('bert-base-uncased')
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        outputs, pooled = model(ms_input_ids)
        
        pt_model = ptBertModel.from_pretrained('bert-base-uncased')
        pt_model.eval()
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        outputs_pt, pooled_pt = pt_model(input_ids=pt_input_ids)

        assert np.allclose(outputs.asnumpy(), outputs_pt.detach().numpy(), atol=1e-5)
        assert np.allclose(pooled.asnumpy(), pooled_pt.detach().numpy(), atol=1e-5)