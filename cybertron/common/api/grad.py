"""
grad api.
"""
from mindspore.ops import stop_gradient, GradOperation

grad_func = GradOperation(True, False, False)
grad_cell = GradOperation(False, True, False)
def value_and_grad(func, params=None, has_aux=False):
    """compute values and grads of function."""
    if params is None:
        grad_ = grad_func
    else:
        grad_ = grad_cell

    def fn_aux(*args):
        outputs = func(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = func

    def value_and_grad_f(*args):
        values = fn_(*args)
        if params is None:
            grads = grad_(fn_)(*args)
        else:
            grads = grad_(fn_, params)(*args)
        return values, grads
    return value_and_grad_f

def grad(func, params=None, has_aux=False):
    """compute grad of function."""
    value_and_grad_f = value_and_grad(func, params, has_aux)
    def grad_f(*args):
        _, grads = value_and_grad_f(*args)
        return grads
    return grad_f
