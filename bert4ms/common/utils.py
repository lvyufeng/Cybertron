import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor

class Erf(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, inputs):
        inputs_dtype = inputs.dtype
        intermidiate = 1.0 / (1.0 + 0.5 * ops.absolute(inputs))
        ans = 1 - intermidiate * ops.exp(ops.neg_tensor(ops.pows(inputs, 2)) - Tensor(1.26551223, inputs_dtype) +
                                         intermidiate * (Tensor(1.00002368, inputs_dtype) +
                                         intermidiate * (Tensor(0.37409196, inputs_dtype) +
                                         intermidiate * (Tensor(0.09678418, inputs_dtype) +
                                         intermidiate * (Tensor(-0.18628806, inputs_dtype) +
                                         intermidiate * (Tensor(0.27886807, inputs_dtype) +
                                         intermidiate * (Tensor(-1.13520398, inputs_dtype) +
                                         intermidiate * (Tensor(1.48851587, inputs_dtype) +
                                         intermidiate * (Tensor(-0.82215223, inputs_dtype) +
                                         intermidiate * (Tensor(0.17087277, inputs_dtype)
                                        ))))))))))
        cond = ops.GreaterEqual()(inputs, 0.0)

        result = ops.Select()(cond, ans, -ans)
        return result
