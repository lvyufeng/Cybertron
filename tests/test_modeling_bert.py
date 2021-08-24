import unittest
import numpy as np
from bert4ms.models import BertModel, BertConfig, BertForPretraining
from mindspore import Tensor
from mindspore import context
import mindspore

class TestModelingBert(unittest.TestCase):
    def test_modeling_bert_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = BertConfig()
        model = BertModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        segment_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        # model.compile((input_ids, segment_ids))
        outputs, pooled = model(input_ids, segment_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = BertConfig()
        model = BertModel(config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        segment_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        model.compile(input_ids, segment_ids)
        outputs, pooled = model(input_ids, segment_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_pretraining_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = BertConfig()
        model = BertForPretraining(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        segment_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        # model.compile((input_ids, segment_ids))
        mlm_logits, nsp_logits = model(input_ids, segment_ids)
        assert mlm_logits.shape == (1, 512, 32000)
        assert nsp_logits.shape == (1, 2)

    def test_modeling_bert_pretraining_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = BertConfig()
        model = BertForPretraining(config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        segment_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        model.compile(input_ids, segment_ids)
        mlm_logits, nsp_logits = model(input_ids, segment_ids)
        assert mlm_logits.shape == (1, 512, 32000)
        assert nsp_logits.shape == (1, 2)