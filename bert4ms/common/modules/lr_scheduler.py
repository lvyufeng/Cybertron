import math
import mindspore
import mindspore.log as logger
import mindspore.ops as ops
from mindspore import ms_class, Parameter, Tensor
from mindspore.ops import constexpr

@constexpr
def log_warning(info):
    logger.warning(info)

@ms_class
class _LRSchedule():
    """ Parent of all LRSchedules here. """
    warn_t_total = False # is set to True for schedules where progressing beyond t_total steps doesn't make sense
    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning("t_total value of {} results in schedule not being applied".format(t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = Parameter(Tensor(-1., mindspore.float32), 'warned_for_t_total_at_progress')

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.
        progress = step / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            log_warning(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly."
                    .format(ret, self.__class__.__name__))
            ops.assign(self.warned_for_t_total_at_progress, progress)
        # end warning
        return ret

    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.

class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    warn_t_total = True
    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1. at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + ops.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)
        assert(cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + ops.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        assert(warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + ops.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """
    warn_t_total = True
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return ops.maximum((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None: ConstantLR,
    "none": ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}