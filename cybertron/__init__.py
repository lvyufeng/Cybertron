try:
    import mindspore
except ModuleNotFoundError as e:
    raise e
else:
    from packaging import version
    mindspore_version = mindspore.__version__
    if version.parse(mindspore_version) < version.parse('1.8.1'):
        raise ValueError(f'The MindSpore version must >= 1.8.1, but get {mindspore_version}')

from .models import *
from .tokenizers import *
from .common import *