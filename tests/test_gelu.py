import unittest
import mindspore
import numpy as np
from mindspore import Tensor, context
from bert4ms.common.activations import GELU

class TestGELU(unittest.TestCase):
    def test_gelu(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        gelu = GELU()
        output = gelu(x)
        result = np.array([[-1.5865526e-01,3.9998732e+00,-0.0000000e+00],
                            [1.9544997e+00,-1.4901161e-06,9.0000000e+00]])
        assert np.allclose(output.asnumpy(), result, atol=1e-5)



