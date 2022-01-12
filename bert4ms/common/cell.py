import logging
import os
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from typing import Optional, Union
from .utils import PRETRAINED_MODEL_ARCHIVE_MAP, load_from_cache
from .config import PretrainedConfig

class PretrainedCell(nn.Cell):
    """"""
    config_class = None
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self):
        pass

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *args, **kwargs):
        """
        Load a pre-trained checkpoint from a pre-trained model file or url,
        download and cache the pre-trained model file if model name in model list. 

        Params:
            pretrained_model_name_or_path:
            cache_dir:
        """
        config = kwargs.pop("config", None)
        # load config
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.load(config_path)

        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            model_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            logging.info("The checkpoint file not found, start to download.")
            model_url = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            model_file = load_from_cache(pretrained_model_name_or_path + '.ckpt', model_url)
        else:
            # Something unknown
            raise ValueError(f"unable to parse {pretrained_model_name_or_path} as a local path or model name")
        # instantiate model
        model = cls(config, *args, **kwargs)
        # load ckpt
        try:
            param_dict = load_checkpoint(model_file)
        except:
            raise ValueError(f"File {model_file} is not a checkpoint file, please check the path.")
        param_not_load = load_param_into_net(model, param_dict)
        if param_not_load:
            raise KeyError("The following weights in model are not found: {param_not_load}")
        
        return model

    def save(self, save_dir: Union[str, os.PathLike]):
        pass

