from ..common.tokenizers import FullTokenizer

class BertTokenizer(FullTokenizer):
    def __init__(self, vocab_file, do_lower_case):
        super().__init__(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, ):
        return None