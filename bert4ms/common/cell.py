import os
import numpy
import inspect
import mindspore.nn as nn
import mindspore.log as logger
import mindspore.common.dtype as mstype
from typing import Callable, Optional, Union
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _executor, _pynative_exec

class Cell(nn.Cell):
    """Modify the origin Cell to split compile and run process."""
    def __call__(self, *inputs, **kwargs):
        if self.__class__.construct is nn.Cell.construct:
            logger.warning(f"The '{self.__class__}' does not override the method 'construct', "
                           f"will call the super class(Cell) 'construct'.")
        if kwargs:
            bound_args = inspect.signature(self.construct).bind(*inputs, **kwargs)
            inputs = bound_args.args
            kwargs = bound_args.kwargs
        if context.get_context("mode") == context.GRAPH_MODE:
            self._check_construct_args(*inputs, **kwargs)
            if self.enable_hook:
                raise ValueError("The graph mode does not support hook function.")
            out = self.run_graph(*inputs)
            return out

        self.do_parameter_broadcast()
        for item in inputs:
            if isinstance(item, numpy.ndarray):
                raise TypeError("cell inputs should not be numpy array.")
        if self.requires_grad is True:
            _pynative_exec.set_grad_flag(True)
        _pynative_exec.new_graph(self, *inputs, **kwargs)
        cast_inputs = list()
        if hasattr(self, "_mindspore_flags"):
            if self._mindspore_flags.get('fp16'):
                cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float16)
            if self._mindspore_flags.get('fp32'):
                cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float32)
        if not cast_inputs:
            cast_inputs = inputs
        output = self.run_construct(cast_inputs, kwargs)
        if isinstance(output, Parameter):
            output = output.data
        _pynative_exec.end_graph(self, output, *inputs, **kwargs)
        return output

    def run_graph(self, *inputs):
        """
        Compiles and runs cell.

        Args:
            inputs (tuple): Inputs of the Cell object.

        Returns:
            Object, the result of executing.
        """
        self._auto_parallel_compile_and_run = True

        new_inputs = []
        for i in inputs:
            if isinstance(i, Tensor):
                new_inputs.append(i)
            elif context.get_context("grad_for_scalar") and isinstance(i, (int, float)):
                new_inputs.append(i)

        if self._auto_parallel_mode:
            if new_inputs and isinstance(new_inputs[0], Tensor) and inputs[0].virtual_flag:
                # get parallel inputs in sink mode, parallel inputs set in _executor.compile
                parallel_inputs_run = self._parallel_inputs_run
            else:
                parallel_inputs_run = new_inputs
            return _executor(self, *parallel_inputs_run, phase=self.phase)
        return _executor(self, *new_inputs, phase=self.phase)

    def apply(self, fn: Callable[['Cell'], None]):
        for cell in self.cells():
            cell.apply(fn)
        fn(self)
        return self

def compile_model(model, inputs):
    model.compile(inputs)

class PretrainedCell(Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]):
        pass

    def save(self, save_dir: Union[str, os.PathLike]):
        pass

