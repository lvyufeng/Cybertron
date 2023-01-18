import unittest
import pytest
import mindspore as ms
import numpy as np
from ddt import ddt, data
from cybertron.models import XLNetModel, XLNetConfig
from mindspore import Tensor

@ddt
class TestModelingXLNet(unittest.TestCase):
    @data(True, False)
    def test_modeling_xlnet(self, mode):
        config = XLNetConfig(n_layer=2, n_head=4)
        model = XLNetModel(config)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            (outputs,) = model(input_ids)
            return outputs

        if mode:
            forward = ms.jit(forward)


        outputs = forward(input_ids)
        assert outputs.shape == (1, 512, 1024)

    @pytest.mark.local
    def test_modeling_xlnet_from_torch(self):
        model = XLNetModel.load('xlnet-base-cased', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        (outputs,) = model(input_ids)
        assert outputs.shape == (1, 512, 768)
