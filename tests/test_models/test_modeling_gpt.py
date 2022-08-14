import unittest
import pytest
import mindspore
import torch
import numpy as np
from cybertron.models import GPTModel, GPTConfig
from mindspore import Tensor
from mindspore import context
from transformers import OpenAIGPTModel

class TestModelingGPT(unittest.TestCase):
    @pytest.mark.action
    def test_modeling_gpt_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        config = GPTConfig()
        model = GPTModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, ) = model(input_ids)
        assert outputs.shape == (1, 512, 768)

    @pytest.mark.action
    def test_modeling_gpt_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        config = GPTConfig()
        model = GPTModel(config)
        model.set_train()
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, ) = model(input_ids)
        assert outputs.shape == (1, 512, 768)

    def test_modeling_gpt_from_torch(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = GPTModel.load('openai-gpt', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        (outputs, ) = model(input_ids)
        assert outputs.shape == (1, 512, 768)

    def test_modeling_gpt_with_ckpt_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        model = GPTModel.load('openai-gpt')
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        (outputs, ) = model(ms_input_ids)
        
        pt_model = OpenAIGPTModel.from_pretrained('openai-gpt')
        pt_model.eval()
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        (outputs_pt, ) = pt_model(input_ids=pt_input_ids)

        assert (outputs.asnumpy() - outputs_pt.detach().numpy()).mean() < 1e-5
