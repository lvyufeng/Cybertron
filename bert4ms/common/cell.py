import os
import numpy
import inspect
import mindspore.nn as nn
import mindspore.log as logger
import mindspore.common.dtype as mstype
from typing import Callable, Optional, Union
from bert4ms.converter.download import load_cache

class PretrainedCell(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_weights(self):
        pass

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], cache_dir=None, from_huggingface=False):
        """
        Load a pre-trained checkpoint from a pre-trained model file or url,
        download and cache the pre-trained model file if model name in model list. 

        Params:
            pretrained_model_name_or_path:
            cache_dir:
        """
        pass
        # load config

        # instantiate model

        # load ckpt or download from huggingface

        return None

    def save(self, save_dir: Union[str, os.PathLike]):
        pass

