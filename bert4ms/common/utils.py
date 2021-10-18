import mindspore
from mindspore import Tensor
import mindspore.ops as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from .cell import Cell

class MaskedFill(nn.Cell):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.select = P.Select()
        self.fill = P.Fill()
        self.cast = P.Cast()
    def construct(self, inputs:Tensor, mask:Tensor):
        mask = self.cast(mask, mstype.bool_)
        masked_value = self.fill(mindspore.float32, inputs.shape, self.value)
        output = self.select(mask, masked_value, inputs)
        return output