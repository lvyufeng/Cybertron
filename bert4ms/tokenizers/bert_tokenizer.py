from ..common.tokenizer import PreTrainedTokenizer, FullTokenizer, WordpieceTokenizer

PRETRAINED_VOCAB_FILES = {
    'bert-base-uncased': "https://sharelist-lv.herokuapp.com/models/bert/bert-base-uncased/vocab.txt?preview",
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
    'bert-base-german-cased': 512,
    'bert-large-uncased-whole-word-masking': 512,
    'bert-large-cased-whole-word-masking': 512,
    'bert-large-uncased-whole-word-masking-finetuned-squad': 512,
    'bert-large-cased-whole-word-masking-finetuned-squad': 512,
    'bert-base-cased-finetuned-mrpc': 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    'bert-base-uncased': {'do_lower_case': True},
    'bert-large-uncased': {'do_lower_case': True},
    'bert-base-cased': {'do_lower_case': False},
    'bert-large-cased': {'do_lower_case': False},
    'bert-base-multilingual-uncased': {'do_lower_case': True},
    'bert-base-multilingual-cased': {'do_lower_case': False},
    'bert-base-chinese': {'do_lower_case': False},
    'bert-base-german-cased': {'do_lower_case': False},
    'bert-large-uncased-whole-word-masking': {'do_lower_case': True},
    'bert-large-cased-whole-word-masking': {'do_lower_case': False},
    'bert-large-uncased-whole-word-masking-finetuned-squad': {'do_lower_case': True},
    'bert-large-cased-whole-word-masking-finetuned-squad': {'do_lower_case': False},
    'bert-base-cased-finetuned-mrpc': {'do_lower_case': False},
}

class BertTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, do_lower_case):
        super().__init__(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, ):
        return None