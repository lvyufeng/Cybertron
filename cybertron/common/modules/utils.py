import mindspore.nn as nn

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(False),
    'gelu_approximate': nn.GELU(),
    'swish':nn.SiLU()
}