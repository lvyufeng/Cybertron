import unittest
import mindspore
import mindspore.nn as nn
from cybertron import BertAdam
from cybertron.common.modules.lr_scheduler import ConstantLR, WarmupLinearSchedule
from cybertron.common.api import grad

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

class OptimizationTest(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_adam(self):
        w = mindspore.Parameter(mindspore.Tensor([0.1, -0.2, -0.1], mindspore.float32))
        target = mindspore.Tensor([0.4, 0.2, -0.5], mindspore.float32)
        criterion = nn.MSELoss()
        grad_fn = grad(criterion)
        # No warmup, constant schedule, no gradient clipping
        optimizer = BertAdam(params=[w], lr=2e-1,
                                          weight_decay=0.0,
                                          max_grad_norm=-1)
        for _ in range(200):
            # loss = criterion(mindspore.Tensor(w), target)
            grads = grad_fn(mindspore.Tensor(w), target)
            optimizer(grads)
        self.assertListAlmostEqual(w.asnumpy(), [0.4, 0.2, -0.5], tol=1e-2)