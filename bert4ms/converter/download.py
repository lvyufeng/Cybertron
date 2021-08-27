import os
import requests
import logging
from pathlib import Path
from tqdm import tqdm
from typing import IO, Union

CACHE_DIR = Path.home() / '.bert4ms'

HUGGINGFACE_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-base-multilingual': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

HUGGINGFACE_PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-base-multilingual': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}

TORCH_CONFIG_NAME = 'bert_config.json'
TORCH_WEIGHTS_NAME = 'pytorch_model.bin'

def load_cache(url_or_filename: Union[str, Path], cache_dir=None):
    if cache_dir is None:
        cache_dir = CACHE_DIR
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    
def download_huggingface_model(model_name, save_path):
    save_path = os.path.realpath(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    url = HUGGINGFACE_PRETRAINED_MODEL_ARCHIVE_MAP.get(model_name, None)
    if url is None:
        logging.error('unsupported model.')
    logging.info(f'downloading model `{model_name}`')

    save_file_name = os.path.join(save_path, f'{model_name}.tar.gz')

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
