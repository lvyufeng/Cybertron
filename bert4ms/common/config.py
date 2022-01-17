import os
import json
import logging
from .utils import load_from_cache

class PretrainedConfig:
    pretrained_config_archive = {}
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)

    @classmethod
    def load(cls, pretrained_model_name_or_path, **kwargs):
        force_download = kwargs.pop('force_download', False)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            config_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in cls.pretrained_config_archive:
            logging.info("The checkpoint file not found, start to download.")
            config_url = cls.pretrained_config_archive[pretrained_model_name_or_path]
            config_file = load_from_cache(pretrained_model_name_or_path + '.json', config_url, force_download=force_download)
        else:
            # Something unknown
            raise ValueError(f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        config = cls.from_json(config_file)

        return config

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        config_map = json.loads(text)
        config = cls()
        for k, v in config_map.items():
            setattr(config, k, v)
        return config