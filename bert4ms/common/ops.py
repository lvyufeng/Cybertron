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

def ibnd_jbnd_to_ijbn(a, b):
    # a -> (i, 1, b, n, d)
    a = a.expand_dims(1)
    # b -> (1, j, b, n, d)
    b = b.expand_dims(0)
    out = a * b
    return out.sum(-1)

def ibnd_snd_to_ibns(a, b):
    # a -> (i, b, 1, n, d)
    a = a.expand_dims(2)
    # b -> (1, 1, s, n, d)
    b = b.expand_dims(0).expand_dims(0)
    out = (a * b).sum(-1)
    return out.swapaxes(2, 3)

def ijbs_ibns_to_ijbn(a, b):
    # a -> (i, j, b, 1, s)
    a = a.expand_dims(3)
    # b -> (i, 1, b, n, s)
    b = b.expand_dims(1)
    out = a * b
    return out.sum(-1)

def ijbn_jbnd_to_ibnd(a, b):
    # a -> (i, j, b, n, 1)
    a = a.expand_dims(-1)
    # b -> (1, j, b, n, d)
    b = b.expand_dims(0)
    out = a * b
    return out.sum(1)

def mbnd_mlb_to_lbnd(a, b):
    # a -> (m, 1, b, n, d)
    a = a.expand_dims(1)
    # b -> (m, l, b, 1, 1)
    b = b.expand_dims(-1).expand_dims(-1)
    out = a * b
    return out.sum(0)

def lbnd_mlb_to_mbnd(a, b):
    # a -> (1, l, b, n, d)
    a = a.expand_dims(0)
    # b -> (m, l, b, 1, 1)
    b = b.expand_dims(-1).expand_dims(-1)
    out = a * b
    return out.sum(1)

def blh_bl_to_bh(a, b):
    b = b.expand_dims(-1)
    out = a * b
    return out.sum(1)

def log_softmax(input, axis=-1):
    return ops.LogSoftmax(axis)(input)

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    return nll_loss(log_softmax(input, 1), target, weight, ignore_index, reduction, label_smoothing)

def nll_loss(input, target, weight=None, ignore_index=None, reduction='mean', label_smoothing=0.0):
    ndim = input.ndim
    if ndim == 2:
        ret = _nll_loss(input, target, -1, weight, ignore_index, reduction)
    elif input.ndim == 4:
        ret = _nll_loss(input, target, 1, weight, ignore_index, reduction)
    else:
        # ndim == 3 or ndim > 4
        n = input.shape[0]
        c = input.shape[1]
        out_size = (n,) + input.shape[2:]
        input = input.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(input, target, 1, weight, ignore_index, reduction)
        else:
            ret = _nll_loss(input, target, 1, weight, ignore_index)
            ret = ret.view(out_size)
    return ret

def _nll_loss(input, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    if target.ndim == input.ndim - 1:
        target = target.expand_dims(target_dim)
    nll_loss = -ops.gather_d(input, target_dim, target)
    smooth_loss = -input.sum(axis=target_dim, keepdims=True)
    if weight is not None:
        loss_weights = ops.gather(weight, target, 0)
        nll_loss = nll_loss * loss_weights
    else:
        loss_weights = ops.ones_like(nll_loss)
    if ignore_index is not None:
        non_pad_mask = ops.equal(target, ignore_index)
        nll_loss = nll_loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(target_dim)
        smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        nll_loss = nll_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.sum() / loss_weights.sum()

    eps_i = label_smoothing / input.shape[target_dim]
    loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss
    return loss