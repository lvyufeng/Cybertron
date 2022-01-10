import unittest

from mindspore.ops.primitive import constexpr
from bert4ms.common.utils import Erf
import mindspore
import numpy as np
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)

class TestErf(unittest.TestCase):
    def test_erf(self):
        x = Tensor([-1, 0, 1, 2, 3], mindspore.float32)
        erf = Erf()
        output = erf(x)
        print(output)
        expected = np.array([-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765], np.float32)
        assert np.allclose(expected, output.asnumpy(), 1e-4, 1e-4)
    
    def test_erf_fp16(self):
        x = Tensor([-1, 0, 1, 2, 3], mindspore.float16)
        erf = Erf()
        output = erf(x)
        print(output)
        expected = np.array([-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765], np.float16)
        assert np.allclose(expected, output.asnumpy(), 1e-3, 1e-3)