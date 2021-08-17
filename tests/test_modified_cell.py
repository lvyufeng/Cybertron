import unittest
import mindspore
import numpy as np
import mindspore.nn as nn
from bert4ms import Cell, compile_model
from mindspore import Tensor

class TestModifiedCell(unittest.TestCase):
    def test_cell_compile(self):
        class Net(Cell):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.dense = nn.Dense(input_size, output_size)
            
            def construct(self, inputs):
                outputs = self.dense(inputs)
                return outputs

        net = Net(100, 1)
        inputs = Tensor(np.random.randn(10, 100), mindspore.float32)
        compile_model(net, inputs)
        output = net(inputs)

        assert output.shape == (10, 1)