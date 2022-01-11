import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor


class Erf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.v1 = Tensor(1.26551223, mstype.float32)
        self.v2 = Tensor(1.00002368, mstype.float32)
        self.v3 = Tensor(0.37409196, mstype.float32)
        self.v4 = Tensor(0.09678418, mstype.float32)
        self.v5 = Tensor(-0.18628806, mstype.float32)
        self.v6 = Tensor(0.27886807, mstype.float32)
        self.v7 = Tensor(-1.13520398, mstype.float32)
        self.v8 = Tensor(1.48851587, mstype.float32)
        self.v9 = Tensor(-0.82215223, mstype.float32)
        self.v10 = Tensor(0.17087277, mstype.float32)

    def construct(self, inputs):
        inputs_dtype = inputs.dtype
        intermidiate = 1.0 / (1.0 + 0.5 * ops.absolute(inputs))
        ans = 1 - intermidiate * ops.exp(ops.neg_tensor(ops.pows(inputs, 2)) - self.v1.astype(inputs_dtype) +
                                         intermidiate * (self.v2.astype(inputs_dtype) +
                                         intermidiate * (self.v3.astype(inputs_dtype) +
                                         intermidiate * (self.v4.astype(inputs_dtype) +
                                         intermidiate * (self.v5.astype(inputs_dtype) +
                                         intermidiate * (self.v6.astype(inputs_dtype) +
                                         intermidiate * (self.v7.astype(inputs_dtype) +
                                         intermidiate * (self.v8.astype(inputs_dtype) +
                                         intermidiate * (self.v9.astype(inputs_dtype) +
                                         intermidiate * (self.v10.astype(inputs_dtype)))))))))))
        cond = ops.GreaterEqual()(inputs, 0.0)

        result = ops.Select()(cond, ans, -ans)
        return result
