import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context
from .ops import Erf

class GELU(nn.Cell):
    def __init__(self, approximate=True):
        """Initialize GELU."""
        super(GELU, self).__init__()
        self.approximate = approximate
        if self.approximate:
            self.gelu = ops.GeLU()
        else:
            if context.get_context("device_target") == "CPU":
                self.erf = Erf()
            else:
                self.erf = ops.Erf()
            self.sqrt = ops.Sqrt()
            self.const0 = Tensor(0.5, mstype.float32)
            self.const1 = Tensor(1.0, mstype.float32)
            self.const2 = Tensor(2.0, mstype.float32)

    def construct(self, x):
        if self.approximate:
            return self.gelu(x)
        return x * ops.cast(self.const0, x.dtype) * (ops.cast(self.const1, x.dtype) + \
            self.erf(x / self.sqrt(ops.cast(self.const2, x.dtype))))

activation_map = {
    'relu': nn.ReLU(),
    'gelu': GELU(False),
    'gelu_approximate': GELU()
}