import logging
import os
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from typing import Optional, Union
from .utils import cached_model
from .config import PretrainedConfig

class PretrainedCell(nn.Cell):
    """"""
    pretrained_model_archive = {}
    pytorch_pretrained_model_archive_list = []
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
        force_download = kwargs.pop('force_download', False)
        from_torch = kwargs.pop('from_torch', False)
        # load config
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.load(config_path)

        if from_torch:
            model_file = cached_model(pretrained_model_name_or_path, cls.pytorch_pretrained_model_archive_list,
                                      from_torch, force_download)
        else:
            model_file = cached_model(pretrained_model_name_or_path, cls.pretrained_model_archive,
                                      from_torch, force_download)

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

