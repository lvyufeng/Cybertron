"""
pretrained tokenizer.
"""
# pylint: disable=R0902
# pylint: disable=R0904
# pylint: disable=R0913
# pylint: disable=W1203
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import logging
import six
import mindspore
from ..utils import load_from_cache

logger = logging.getLogger(__name__)

class PreTrainedTokenizer:
    """
    Pretrained Tokenizer.
    """
    pretrained_vocab = {}
    max_model_input_sizes = {}
    pretrained_init_configuration = {}

    SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
                                 "pad_token", "cls_token", "mask_token",
                                 "additional_special_tokens"]
    def __init__(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

        # Added tokens
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading
        # (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) \
                        and all(isinstance(t, str) for t in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

    @classmethod
    def load(cls, pretrained_model_name_or_path, *args, **kwargs):
        """load tokenizer."""
        force_download = kwargs.pop('force_download', False)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            vocab_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in cls.pretrained_vocab:
            logging.info("The checkpoint file not found, start to download.")
            vocab_url = cls.pretrained_vocab[pretrained_model_name_or_path]
            vocab_file = load_from_cache(pretrained_model_name_or_path + '.txt',
                                         vocab_url, force_download=force_download)
        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            kwargs.update(cls.pretrained_init_configuration[pretrained_model_name_or_path])
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer
        tokenizer = cls(vocab_file, *args, **kwargs)
        return tokenizer

    @property
    def bos_token(self):
        """ Beginning of sentence token (string).
        Log an error if used while not having been set.
        """
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        """ Separation token (string). E.g. separate context and query in an input sequence.
        Log an error if used while not having been set. """
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging
        self-attention along the full depth of the model.
        Log an error if used while not having been set. """
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling.
        Log an error if used while not having been set. """
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings).
        Log an error if used while not having been set. """
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self._additional_special_tokens

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary.
        E.g. separate context and query in an input sequence.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary.
        E.g. to extract a summary of an input sequence leveraging self-attention along
        the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary.
        E.g. when training a model with masked-language modeling.
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers).
        Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    def vocab_size(self):
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError


    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)


    def add_tokens(self, new_tokens):
        """
        Add a list of new tokens to the tokenizer class.
        """
        if not new_tokens:
            return 0

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str)
            if token != self.unk_token and \
                    self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token):
                to_add_tokens.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v:k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def num_added_tokens(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.
        """

        if pair:
            initial_tokens_len = len(self.encode("This is a sequence") + \
                self.encode("This is another"))
            final_tokens_len = len(self.encode("This is a sequence",
                                               "This is another", add_special_tokens=True))
        else:
            initial_tokens_len = len(self.encode("This is a sequence"))
            final_tokens_len = len(self.encode("This is a sequence", add_special_tokens=True))

        return final_tokens_len - initial_tokens_len

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes.
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str)
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).
            Take care of added tokens.
        """
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text, **kwargs)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in self.all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return sum((self._tokenize(token, **kwargs) if token not \
                    in self.added_tokens_encoder and token not in self.all_special_tokens \
                    else [token] for token in tokenized_text), [])

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).
            Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            logger.warning(
                f"Token indices sequence length is longer than the specified "
                f"maximum sequence length for this model ({len(ids)} > {self.max_len})."
                f"Running this sequence through the model will result in indexing errors")
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(self,
                text,
                text_pair=None,
                add_special_tokens=False,
                max_length=None,
                stride=0,
                truncate_first_sequence=True,
                return_tensors=None,
                **kwargs):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.
        """
        encoded_inputs = self.encode_plus(text,
                                          text_pair=text_pair,
                                          max_length=max_length,
                                          add_special_tokens=add_special_tokens,
                                          stride=stride,
                                          truncate_first_sequence=truncate_first_sequence,
                                          return_tensors=return_tensors,
                                          **kwargs)

        return encoded_inputs["input_ids"]

    def encode_plus(self,
                    text,
                    text_pair=None,
                    add_special_tokens=False,
                    max_length=None,
                    stride=0,
                    truncate_first_sequence=True,
                    return_tensors=None,
                    **kwargs):
        """
        Returns a dictionary containing the encoded sequence or
        sequence pair and additional informations.
        """

        def get_input_ids(text):
            if isinstance(text, six.string_types):
                return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
            if isinstance(text, (list, tuple)) and len(text) > 0 \
                and isinstance(text[0], six.string_types):
                return self.convert_tokens_to_ids(text)
            if isinstance(text, (list, tuple)) and len(text) > 0 \
                and isinstance(text[0], int):
                return text
            raise ValueError("Input is not valid. Should be a string, "
                             "a list/tuple of strings or a list/tuple of integers.")

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(first_ids,
                                      pair_ids=second_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncate_first_sequence=truncate_first_sequence,
                                      return_tensors=return_tensors)


    def prepare_for_model(self, ids, pair_ids=None, max_length=None,
                          add_special_tokens=False, stride=0,
                          truncate_first_sequence=True, return_tensors=False):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        if max_length:
            n_added_tokens = self.num_added_tokens(pair=pair) if add_special_tokens else 0
            if pair and n_added_tokens + \
                (len_pair_ids if truncate_first_sequence else len_ids) >= max_length:
                logger.warning(
                    "You supplied a pair of sequence in which the sequence that "
                    "will not be truncated is longer than the maximum specified length."
                    "This pair of sequences will not be truncated.")
            else:
                if n_added_tokens + len_ids + len_pair_ids > max_length:
                    if truncate_first_sequence or not pair:
                        encoded_inputs["overflowing_tokens"] = \
                            ids[max_length - len_pair_ids - n_added_tokens - stride:]
                        ids = ids[:max_length - len_pair_ids - n_added_tokens]
                    elif not truncate_first_sequence and pair:
                        encoded_inputs["overflowing_tokens"] = \
                            pair_ids[max_length - len_ids - n_added_tokens - stride:]
                        pair_ids = pair_ids[:max_length - len_ids - n_added_tokens]
                    else:
                        logger.warning(
                            "Cannot truncate second sequence as it is not provided. No truncation.")

        if add_special_tokens:
            sequence = self.add_special_tokens_sequence_pair(ids, pair_ids) \
                if pair else self.add_special_tokens_single_sequence(ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids) \
                if pair else [0] * len(sequence)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_tensors:
            sequence = mindspore.Tensor([sequence], mindspore.int32)
            token_type_ids = mindspore.Tensor([token_type_ids], mindspore.int32)
        elif return_tensors is not None:
            logger.warning(f"Unable to convert output to tensors format {return_tensors}, "
                           f"PyTorch or TensorFlow is not available.")

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids

        return encoded_inputs

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
        """create token type ids from sequences"""
        logger.warning("This tokenizer does not make use of special tokens.")
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def add_special_tokens_single_sequence(self, token_ids):
        """add special tokens single sequence"""
        logger.warning("This tokenizer does not make use of special tokens. "
                       "The sequence has been returned with no modification.")
        return token_ids

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """add special tokens sequence pair."""
        logger.warning("This tokenizer does not make use of special tokens. "
                       "The two sequences have been concatenated.")
        return token_ids_0 + token_ids_1

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.
            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens).
                                     Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
        return ' '.join(self.convert_ids_to_tokens(tokens))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids,
                                                     skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(" " + token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ''.join(sub_texts)

        if self._sep_token is not None and self._sep_token in text:
            text = text.replace(self._cls_token, self._sep_token)
            split_text = list(filter(lambda sentence: len(sentence) > 0,
                                     text.split(self._sep_token)))
            if clean_up_tokenization_spaces:
                clean_text = [self.clean_up_tokenization(text) for text in split_text]
                return clean_text
            return split_text
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + \
                (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = list(self._convert_token_to_id(t) for t in all_toks)
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces
         before punctuations and abreviated forms.
        """
        out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!') \
                            .replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't") \
                            .replace(" 'm", "'m").replace(" do not", " don't") \
                            .replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        return out_string
