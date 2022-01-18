import io
import collections
import unicodedata
import six
import os
import logging
from typing import Callable, Optional, Union
from .utils import load_from_cache

class PreTrainedTokenizer:
    pretrained_vocab = {}
    max_model_input_sizes = {}
    pretrained_init_configuration = {}

    def __init__(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

    @classmethod
    def load(cls, pretrained_model_name_or_path, *args, **kwargs):
        force_download = kwargs.pop('force_download', False)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            vocab_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in cls.pretrained_vocab:
            logging.info("The checkpoint file not found, start to download.")
            vocab_url = cls.pretrained_vocab[pretrained_model_name_or_path]
            vocab_file = load_from_cache(pretrained_model_name_or_path + '.txt', vocab_url, force_download=force_download)
        else:
            # Something unknown
            raise ValueError(f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            kwargs.extend(cls.pretrained_init_configuration[pretrained_model_name_or_path])
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer
        tokenizer = cls(vocab_file, *args, **kwargs)
        return tokenizer