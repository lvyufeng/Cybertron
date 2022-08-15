import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import constexpr
# from mindspore.ops._primitive_cache import _get_cache_prim


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

@constexpr
def raise_value_error(info):
    raise ValueError(info)

@constexpr
def raise_runtime_error(info):
    raise RuntimeError(info)

@constexpr
def raise_type_error(info):
    raise TypeError(info)

inf = float('inf')

def _check_dtype(d1, d2):
    if mindspore.float32 in (d1, d2):
        return mindspore.float32
    if d1 == d2:
        return d1
    raise ValueError('dtype is not supported.')

def dot(a, b):
    res_dtype = _check_dtype(a.dtype, b.dtype)
    ndim_a, ndim_b = a.ndim, b.ndim
    if ndim_a == 0 or ndim_b == 0:
        return ops.tensor_mul(a, b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = ops.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = ops.transpose(b, perm)

    if a.shape[-1] != b.shape[-1]:
        raise_value_error('shapes are not aligned')
    a_aligned = a.reshape(-1, a.shape[-1]).astype(mindspore.float32)
    b_aligned = b.reshape(-1, b.shape[-1]).astype(mindspore.float32)

    res = ops.matmul(a_aligned, b_aligned.T)
    res = res.reshape(a.shape[:-1] + b.shape[:-1])

    return res.astype(res_dtype)

def sqrt(x):
    return ops.sqrt(x.astype(mindspore.float32))

_reciprocal = ops.Reciprocal()
def reciprocal(x):
    if isinstance(x, Tensor):
        # reciprocal = _get_cache_prim(ops.Reciprocal)()
        return _reciprocal(x)
    return 1/x

# grad operations
def get_grads():
    pass

def bmm(x, y, transpose_x=False, transpose_y=False):
    # return _get_cache_prim(ops.BatchMatMul)(transpose_x, transpose_y)(x, y)
    return ops.BatchMatMul(transpose_x, transpose_y)(x, y)

def masked_fill_(inputs:Tensor, mask:Tensor, value:float):
    # return ops.masked_fill(inputs, mask, value)
    return input


@constexpr
def _check_axis(axis, ord, ndim):
    if axis is None:
        axis = tuple(range(ndim))
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)):
            return axis, True
        else:
            return axis, False
    else:
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, tuple):
            if len(axis) > 2:
                raise ValueError("Improper number of dimensions to norm.")
        else:
            raise ValueError(f'axis should be int or tuple but got {type(axis)}')
        return axis, False

@constexpr
def _check_ord(ord, axis):
    if len(axis) == 1:
        if isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
    elif len(axis) == 2:
        if ord not in [2, -2, 1, -1, inf, -inf, 'fro', 'f', 'nuc', None]:
            raise ValueError("Invalid norm order for matrices.")

def norm(x, ord=None, axis=None, keepdims=False):
    ndim = x.ndim
    # Normalize the `axis` argument to a tuple.
    axis, immediate = _check_axis(axis, ord, ndim)
    _check_ord(ord, axis)
    # Immediately handle some default, simple, fast, and common cases.
    if immediate:
        x = x.ravel()
        sqnorm = dot(x, x)
        ret = sqrt(sqnorm)
        if keepdims:
            ret = ret.reshape(ndim*[1])
        return ret

    if isinstance(ord, int):
        return ops.LpNorm(axis, ord, keepdims)

    if len(axis) == 1:
        if ord == inf:
            return ops.abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -inf:
            return ops.abs(x).min(axis=axis, keepdims=keepdims)
        elif ord is None:
            # special case for speedup
            # conj = _get_cache_prim(ops.Conj)()
            conj = ops.Conj()
            s = conj(x) * x
            # reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
            reduce_sum = ops.ReduceSum(keepdims)
            return sqrt(reduce_sum(s, axis=axis))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        else:
            absx = ops.abs(x)
            absx **= ord
            # reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
            reduce_sum = ops.ReduceSum(keepdims)
            ret = reduce_sum(absx, axis=axis)
            ret **= reciprocal(ord)
            if ops.isnan(ret):
                return ops.zeros_like(ret)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, ndim)
        col_axis = normalize_axis_index(col_axis, ndim)
        if row_axis == col_axis:
            raise_value_error('Duplicate axes given.')

    
        if ord == inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in ['fro', 'f']:
            # conj = _get_cache_prim(ops.Conj)()
            conj = ops.Conj()
            ret = sqrt(ops.reduce_sum((conj(x) * x), axis=axis))
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            conj = ops.Conj()
            ret = sqrt(ops.reduce_sum((conj(x) * x), axis=axis))
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        return None

def _multi_svd_norm(x, row_axis, col_axis, op):
    y = moveaxis(x.astype(mindspore.float32), (row_axis, col_axis), (-2, -1))
    if op == 'amax':
        result = ops.svd(y, compute_uv=False).max(axis=-1)
    elif op == 'amin':
        result = ops.svd(y, compute_uv=False).min(axis=-1)
    else:
        result = None
    return result

def normalize_axis_index(axis, ndim):
    if axis >= 0 and axis < ndim:
        return axis
    elif axis < 0 and axis >= -ndim:
        return ndim + axis
    else:
        raise_value_error('axis is out of range.')
        return None

def moveaxis(x, source, destination):
    perm = [i for i in range(x.ndim)]
    for s, d in zip(source, destination):
        tmp = perm[s]
        perm[s] = perm[d]
        perm[d] = tmp
    perm = tuple(perm)
    return ops.transpose(x, perm)

def clip_grad_norm(grads, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False):
    if isinstance(grads, mindspore.Tensor):
        grads = [grads]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return [], mindspore.Tensor(0., mindspore.float32)

    if norm_type == inf:
        norms = [grad.abs().max() for grad in grads]
        total_norm = norms[0] if len(norms) == 1 else ops.max(ops.stack(norms))
    else:
        norms = ()
        for grad in grads:
            norms += (norm(grad, norm_type),)
        total_norm = norm(ops.stack(norms), norm_type)

    if error_if_nonfinite and ops.logical_or(ops.isnan(total_norm), ops.bool_not(ops.isfinite(total_norm))):
        raise_runtime_error(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = clip_coef.clip(None, 1.0)
    new_grads = []
    for grad in grads:
        new_grads.append(ops.mul(grad, clip_coef_clamped))
    return new_grads, total_norm