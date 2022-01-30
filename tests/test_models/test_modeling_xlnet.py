import unittest
import mindspore
import torch
import numpy as np
from bert4ms.models import XLNetModel, XLNetConfig
from mindspore import Tensor
from mindspore import context
from transformers import XLNetModel as ptXLNetModel

class TestModelingXLNet(unittest.TestCase):
    def test_modeling_xlnet_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = XLNetConfig()
        model = XLNetModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, new_mems) = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert len(new_mems) == config.n_layer

    def test_modeling_xlnet_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = XLNetConfig()
        model = XLNetModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, new_mems) = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert len(new_mems) == config.n_layer

    def test_modeling_xlnet_from_torch(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = XLNetModel.load('xlnet-base-cased', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, _) = model(input_ids)
        assert outputs.shape == (1, 512, 768)

    def test_modeling_xlnet_with_ckpt_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = XLNetModel.load('xlnet-base-cased')
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        (outputs, _) = model(ms_input_ids)
        
        pt_model = ptXLNetModel.from_pretrained('xlnet-base-cased')
        pt_model.eval()
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        (outputs_pt, _) = pt_model(input_ids=pt_input_ids)

        assert (outputs.asnumpy() - outputs_pt.detach().numpy()).mean() < 1e-5