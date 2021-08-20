import mindspore
from mindspore import Tensor
import mindspore.ops as P
from .cell import Cell

class MaskedFill(Cell):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.select = P.Select()
        self.fill = P.Fill()

    def construct(self, inputs:Tensor, mask:Tensor):
        masked_value = self.fill(mindspore.float32, inputs.shape, self.value)
        output = self.select(mask, masked_value, inputs)
        return output