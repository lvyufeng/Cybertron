import unittest
from bert4ms.tokenizers.bert_tokenizer import BertTokenizer

class TestBertTokenizer(unittest.TestCase):
    def test_bert_tokenizer(self):
        tokenizer = BertTokenizer.load('bert-base-uncased')
        inputs = tokenizer('hello world.')

        assert inputs == ['hello', 'world', '.']