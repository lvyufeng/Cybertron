import unittest
from bert4ms.converter.mapper import build_weight_map
from bert4ms.converter import download_huggingface_model

class TestLoadHuggingfaceCKPT(unittest.TestCase):
    def test_build_weight_map(self):
        weight_map = build_weight_map('bert', 12)
        for i in range(12):
            assert 'bert.encoder.layer.{}.attention.self.query.weight'.format(i) in weight_map

    def test_download_huggingface_model(self):
        path = '/root'
        model_name = 'bert-base-uncased'
        download_huggingface_model(model_name, path)
    
    def test_convert_huggingface_model_to_mindspore(self):
        pass

    def test_load_mindspore_ckpt(self):
        pass

    def test_mindspore_model_inference(self):
        pass