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

class SiLU(nn.Cell):
    """Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.
    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    Examples::
        >>> m = nn.SiLU()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        return inputs * self.sigmoid(inputs)

activation_map = {
    'relu': nn.ReLU(),
    'gelu': GELU(False),
    'gelu_approximate': GELU(),
    'swish':SiLU()
}