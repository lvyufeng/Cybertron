import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from collections import OrderedDict
from mindspore import log as logger
from mindspore import Parameter, Tensor, ParameterTuple
from ..utils import required
from .lr_scheduler import _LRSchedule, SCHEDULES
from .ops import clip_grad_norm

class Optimizer(nn.Cell):
    def __init__(self, params, defaults):
        super().__init__()
        self.param_groups = []
        self.defaults = defaults

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    @property    
    def parameters(self):
        flatten_params = []
        for param_group in self.param_groups:
            flatten_params.extend([param for param in param_group[0]])

        return flatten_params
    
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"
        assert 'params' in param_group

        params = param_group['params']
        if isinstance(params, mindspore.Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, mindspore.Parameter):
                raise TypeError("optimizer can only optimize Parameter, "
                                "but one of the params is " + str(type(param)))

        new_param_group = OrderedDict({'params': param_group['params']})
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            elif name in param_group:
                new_param_group.setdefault(name, param_group[name])
            else:
                new_param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            logger.warning("optimizer contains a parameter group with duplicate parameters")

        param_set = set()
        for group in self.param_groups:
            # group[0] is parameter list
            param_set.update(set(group[0]))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append([v for _, v in new_param_group.items()])

class BertAdam(Optimizer):
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule. "
                               "Please specify custom warmup and t_total in _LRSchedule object.")
        defaults = OrderedDict(lr=lr, schedule=schedule,
                               b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                               max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)
        self.state_init()

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for _ in range(len(group[0])):
                lr_scheduled = group[1]
                lr_scheduled *= group[2].get_lr(self.step)
                lr.append(lr_scheduled)
        return lr

    def state_init(self):
        # State initialization
        self.states = []
        self.step = Parameter(Tensor(0), 'step')
        for group in self.param_groups:
            state_list = []
            for p in group[0]:
                next_m = Parameter(Tensor(np.ones(p.shape), p.dtype), p.name + '_next_m')
                next_v = Parameter(Tensor(np.ones(p.shape), p.dtype), p.name + '_next_v')
                state_list.append(ParameterTuple([next_m, next_v]))
            self.states.append(state_list)

    def construct(self, grads):
        count = 0
        for g_idx, group in enumerate(self.param_groups):
            params, lr, schedule, beta1, beta2, e, weight_decay, max_grad_norm = group
            lr_scheduled = schedule(self.step) * lr
            for p_idx, p in enumerate(params):
                state = self.states[g_idx][p_idx]
                next_m, next_v = state[0], state[1]
                grad = grads[count]
                # Add grad clipping
                if max_grad_norm > 0:
                    grad = clip_grad_norm(grad, max_grad_norm)[0][0]
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                ops.assign(next_m, ops.add(ops.mul(next_m, beta1), grad * (1 - beta1)))
                # next_m.mul_(beta1).add_(1 - beta1, grad)
                ops.assign(next_v, ops.add(ops.mul(next_v, beta2), grad * grad * (1 - beta2)))
                update = next_m / (ops.sqrt(next_v) + e)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if weight_decay > 0.0:
                    update += weight_decay * p

                update_with_lr = lr_scheduled * update
                ops.assign(p, p - update_with_lr)

                count += 1
        
                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
        ops.assign(self.step, self.step + 1)
        return True
