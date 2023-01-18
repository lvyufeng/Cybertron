import unittest
import pytest
import mindspore
import torch
import numpy as np
from cybertron.models import XLNetModel, XLNetConfig
from cybertron.utils import convert_state_dict
from mindspore import Tensor
from transformers import XLNetModel as ptXLNetModel
from transformers import XLNetConfig as ptXLNetConfig

class TestXLNetComparison(unittest.TestCase):

    def test_xlnet_comparison(self):
        model = XLNetModel(XLNetConfig(d_model=512,
                                       n_layer=8,
                                       n_head=8,
                                       d_inner=1024,))
        model.set_train(False)
        pt_model = ptXLNetModel(ptXLNetConfig(d_model=512,
                                              n_layer=8,
                                              n_head=8,
                                              d_inner=1024,))
        pt_model.eval()

        ms_dict = convert_state_dict(pt_model, 'xlnet')
        mindspore.load_param_into_net(model, ms_dict)

        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        (outputs,) = model(ms_input_ids)
        
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        (outputs_pt,) = pt_model(input_ids=pt_input_ids)

        assert (outputs.asnumpy() - outputs_pt.detach().numpy()).mean() < 1e-5
