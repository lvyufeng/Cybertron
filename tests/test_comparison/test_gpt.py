import unittest
import mindspore
import torch
from cybertron.models import GPTModel, GPTConfig
from cybertron.utils import convert_state_dict
from mindspore import Tensor
from transformers import OpenAIGPTModel, OpenAIGPTConfig

class TestGPTComparison(unittest.TestCase):
    def test_gpt_comparison(self):
        model = GPTModel(GPTConfig())
        model.set_train(False)
        pt_model = OpenAIGPTModel(OpenAIGPTConfig())
        pt_model.eval()

        ms_dict = convert_state_dict(pt_model, 'gpt')
        mindspore.load_param_into_net(model, ms_dict)

        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        (outputs, ) = model(ms_input_ids)
        
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        (outputs_pt, ) = pt_model(input_ids=pt_input_ids)

        assert (outputs.asnumpy() - outputs_pt.detach().numpy()).mean() < 1e-5
