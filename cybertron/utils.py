import os
import requests
import tempfile
import logging
import shutil
import importlib
from pathlib import Path
from tqdm import tqdm
from typing import IO

try:
    from pathlib import Path
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', Path.home() / '.bert4ms'))
except (AttributeError, ImportError):
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', os.path.join(os.path.expanduser("~"), '.bert4ms')))

CACHE_DIR = Path.home() / '.bert4ms'
HUGGINGFACE_BASE_URL = 'https://huggingface.co/{}/resolve/main/pytorch_model.bin'

def load_from_cache(name, url, cache_dir:str=None, force_download=False):
    """
    Given a URL, load the checkpoint from cache dir if it exists,
    else, download it and save to cache dir and return the path
    """
    if cache_dir is None:
        cache_dir = BERT4MS_CACHE

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    name = name.replace('/', '_')
    cache_path = os.path.join(cache_dir, name)

    # download the checkpoint if not exist
    ckpt_exist = os.path.exists(cache_path)
    if not ckpt_exist or force_download:
        if ckpt_exist:
            os.remove(cache_path)
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

def convert_state_dict(module_or_pth_file, model):
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")

    from mindspore import Tensor, Parameter
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')

    if isinstance(module_or_pth_file, torch.nn.Module):
        is_module = True
        state_dict = module_or_pth_file.state_dict()
    else:
        is_module = False
        state_dict = torch.load(module_or_pth_file, map_location=torch.device('cpu'))

    convert_func = importlib.import_module('cybertron.models.' + model).torch_to_mindspore
    ms_ckpt = convert_func(state_dict)

    if is_module:
        return {i['name']: Parameter(i['data'], i['name']) for i in ms_ckpt}

    ms_ckpt_path = module_or_pth_file.replace('.bin','.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')

    return ms_ckpt_path
