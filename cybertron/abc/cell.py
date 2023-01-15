"""
pretrained cell
"""
import logging
import os
from typing import Optional, Union
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ..utils import load_from_cache, convert_state_dict, HUGGINGFACE_BASE_URL
from .config import PretrainedConfig

class PretrainedCell(nn.Cell):
    """
    Pretrained Cell.

    Args:
        xxx
    """
    name = None
    pretrained_model_archive = {}
    pytorch_pretrained_model_archive_list = []
    config_class = None
 
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_weights(self):
        """init weights of cell."""
        raise NotImplementedError

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
             *args, **kwargs):
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

        # instantiate model
        model = cls(config, *args, **kwargs)

        # download ckpt
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            model_file = os.path.join(pretrained_model_name_or_path)
            assert os.path.isfile(model_file)
        elif pretrained_model_name_or_path in cls.pretrained_model_archive and not from_torch:
            logging.info("The checkpoint file not found, start to download.")
            model_url = cls.pretrained_model_archive[pretrained_model_name_or_path]
            model_file = load_from_cache(pretrained_model_name_or_path + '.ckpt',
                                         model_url,
                                         force_download=force_download)
        elif pretrained_model_name_or_path in cls.pytorch_pretrained_model_archive_list:
            logging.info("The checkpoint file not found in archive list, "
                         "start to download from torch.")
            model_url = HUGGINGFACE_BASE_URL.format(pretrained_model_name_or_path)
            torch_model_file = load_from_cache(pretrained_model_name_or_path + '.bin',
                                               model_url,
                                               force_download=force_download)
            model_file = convert_state_dict(torch_model_file, cls.name)

        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        # load ckpt
        try:
            param_dict = load_checkpoint(model_file)
        except Exception as exc:
            raise ValueError(f"File {model_file} is not a checkpoint file, "
                             f"please check the path.") from exc

        param_not_load = load_param_into_net(model, param_dict)
        if len(param_not_load) == len(model.trainable_params()):
            raise KeyError(f"The following weights in model are not found: {param_not_load}")

        return model

    def save(self, save_dir: Union[str, os.PathLike]):
        """save pretrained cell"""
        raise NotImplementedError
