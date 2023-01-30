import unittest
import pytest
import os
from cybertron.utils import load_from_cache
from cybertron.transformers.bert.bert import CONFIG_ARCHIVE_MAP

class TestLoadFromCache(unittest.TestCase):
    def test_load_from_cache_default_path(self):
        name = 'bert-base-uncased'
        url = CONFIG_ARCHIVE_MAP['bert-base-uncased']
        cache_path = load_from_cache(name + '.json', url)
        assert os.path.exists(cache_path)
    
    def test_load_from_cache_default_path_force_download(self):
        name = 'bert-base-uncased'
        url = CONFIG_ARCHIVE_MAP['bert-base-uncased']
        cache_path = load_from_cache(name + '.json', url, force_download=True)
        assert os.path.exists(cache_path)
