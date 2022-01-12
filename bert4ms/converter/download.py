from genericpath import exists
import os
import requests
import logging
from pathlib import Path
from requests.api import get
from tqdm import tqdm
from typing import IO, Union
from urllib.parse import urlparse

CACHE_DIR = Path.home() / '.bert4ms'

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://111-178-8-193.d.123pan.cn:30443/123-921/cce568a8/1716050-0/cce568a80b72d185e0b5b116b10f0b99?v=1&t=1642040511&s=661221aaa5fb4d5f29d0f970d04ef358&filename=model.ckpt"
}

CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': 'https://sharelist-lv.herokuapp.com/checkpoint/bert-base-uncased/config.json'
}


def http_get(url: str, temp_file:IO):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.writa(chunk)
    progress.close()
