import unittest
import os
from bert4ms.common.utils import load_from_cache
from bert4ms.configs.bert import CONFIG_ARCHIVE_MAP

class TestLoadFromCache(unittest.TestCase):
    def test_load_from_cache_default_path(self):
        name = 'bert-base-uncased'
        url = CONFIG_ARCHIVE_MAP['bert-base-uncased']
        cache_path = load_from_cache(name + '.json', url)
        assert os.path.exists(cache_path)