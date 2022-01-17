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

def load_from_cache(name, url, cache_dir:str=None, force_download=False):
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
    if not os.path.exists(cache_path) or force_download:
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

def convert_state_dict(pth_file):
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")

    from mindspore import Tensor
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file)
    # weight_map = build_weight_map('bert', 12)
    for k, v in state_dict.items():
        if 'embeddings' in k:
            k = k.replace('weight', 'embedding_table')
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
        if 'self' in k:
            k = k.replace('self', 'self_attn')
        print(k)
        ms_ckpt.append({'name': k, 'data':Tensor(v.numpy())})
    ms_ckpt_path = pth_file.replace('.bin','.ckpt')
    try:
        save_checkpoint(ms_ckpt, ms_ckpt_path)
    except:
        raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')
    return ms_ckpt_path

def cached_model(pretrained_model_name_or_path, pretrained_model_archive, from_torch, force_download):
    if os.path.exists(pretrained_model_name_or_path):
        # File exists.
        if not from_torch:
            model_file = os.path.join(pretrained_model_name_or_path, 'model.ckpt')
        else:
            model_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        assert os.path.isfile(model_file)
    elif pretrained_model_name_or_path in pretrained_model_archive:
        logging.info("The checkpoint file not found, start to download.")
        model_url = pretrained_model_archive[pretrained_model_name_or_path]
        if not from_torch:
            model_file = load_from_cache(pretrained_model_name_or_path + '.ckpt', model_url, force_download=force_download)
        else:
            torch_model_file = load_from_cache(pretrained_model_name_or_path + '.bin', model_url, force_download=force_download)
            model_file = convert_state_dict(torch_model_file)
    else:
        # Something unknown
        raise ValueError(f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

    return model_file