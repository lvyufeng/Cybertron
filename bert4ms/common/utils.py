import os
import requests
import tempfile
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import IO

try:
    from pathlib import Path
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', Path.home() / '.bert4ms'))
except (AttributeError, ImportError):
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', os.path.join(os.path.expanduser("~"), '.bert4ms')))

CACHE_DIR = Path.home() / '.bert4ms'

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://sharelist-lv.herokuapp.com/checkpoint/bert-base-uncased/model.ckpt"
}

CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': 'https://sharelist-lv.herokuapp.com/checkpoint/bert-base-uncased/config.json'
}


def load_from_cache(name, url, cache_dir:str=None):
    """
    Given a URL, load the checkpoint from cache dir if it exists,
    else, download it and save to cache dir and return the path
    """
    if cache_dir is None:
        cache_dir = BERT4MS_CACHE

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_path = os.path.join(cache_dir, name)

    # download the checkpoint if not exist
    if not os.path.exists(cache_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            logging.info(f"{name} not found in cache, downloading to {temp_file.name}")

            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)

            logging.info(f"copying {temp_file.name} to cache at {cache_path}")
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    
    return cache_path

def http_get(url: str, temp_file:IO):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()
