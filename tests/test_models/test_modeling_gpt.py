import unittest
import pytest
import mindspore as ms
import numpy as np
from ddt import ddt, data
from cybertron.models import GPTModel, GPTConfig
from mindspore import Tensor
from mindspore import context

@ddt
class TestModelingGPT(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GPTConfig(vocab_size=1000,
                                n_embd=256,
                                n_layer=4,
                                n_head=4)


    @data(True, False)
    def test_modeling_gpt(self, mode):
        model = GPTModel(self.config)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        def forward(input_ids):
            (outputs, ) = model(input_ids)
            return outputs
        
        if mode:
            forward = ms.jit(forward)
        
        outputs = forward(input_ids)
        assert outputs.shape == (1, 512, self.config.n_embd)

    @pytest.mark.local
    def test_modeling_gpt_from_torch(self):
        model = GPTModel.load('openai-gpt', from_torch=True)

        input_ids = Tensor(np.random.randn(1, 512), ms.int32)

        (outputs, ) = model(input_ids)
        assert outputs.shape == (1, 512, 768)
