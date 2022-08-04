import unittest
import mindspore
from bert4ms import BertAdam
from bert4ms.common.modules.lr_scheduler import ConstantLR, WarmupLinearSchedule

class ScheduleInitTest(unittest.TestCase):
    def test_bert_sched_init(self):
        m = mindspore.nn.Dense(50, 50)
        optim = BertAdam(m.trainable_params(), lr=0.001, warmup=.1, t_total=1000, schedule=None)
        self.assertTrue(isinstance(optim.param_groups[0][2], ConstantLR))
        optim = BertAdam(m.trainable_params(), lr=0.001, warmup=.1, t_total=1000, schedule="none")
        self.assertTrue(isinstance(optim.param_groups[0][2], ConstantLR))
        optim = BertAdam(m.trainable_params(), lr=0.001, warmup=.01, t_total=1000)
        self.assertTrue(isinstance(optim.param_groups[0][2], WarmupLinearSchedule))
        # shouldn't fail
