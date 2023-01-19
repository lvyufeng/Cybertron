"""
pretrained cell
"""
import logging
import os
from typing import Optional, Union, Dict
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ..utils import load_from_cache, convert_state_dict, HUGGINGFACE_BASE_URL
from .config import PretrainedConfig

class PretrainedCell(nn.Cell):
    """
    Pretrained Cell.

    Args:
        xxx
    """
    name = None
    pretrained_model_archive = {}
    pytorch_pretrained_model_archive_list = []
    config_class = None
 
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Tie weights if needed
        self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict):
        """ Prunes heads of the base model.
            Arguments:
                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        output_embeddings.embedding_table = input_embeddings.embedding_table

        # if getattr(output_embeddings, "bias", None) is not None:
        #     output_embeddings.bias.data = torch.nn.functional.pad(
        #         output_embeddings.bias.data,
        #         (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
        #         "constant",
        #         0,
        #     )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.
        Arguments:
            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end
        Args:
            old_embeddings: ``torch.nn.Embedding``
                Old embeddings to be resized.
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embedding``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
             *args, **kwargs):
        """
        Load a pre-trained checkpoint from a pre-trained model file or url,
        download and cache the pre-trained model file if model name in model list.

        Params:
            pretrained_model_name_or_path:
            cache_dir:
        """
        config = kwargs.pop("config", None)
        force_download = kwargs.pop('force_download', False)
        from_torch = kwargs.pop('from_torch', False)
        # load config
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.load(config_path)

        # instantiate model
        model = cls(config, *args, **kwargs)

        # download ckpt
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            model_file = os.path.join(pretrained_model_name_or_path)
            assert os.path.isfile(model_file)
        elif pretrained_model_name_or_path in cls.pretrained_model_archive and not from_torch:
            logging.info("The checkpoint file not found, start to download.")
            model_url = cls.pretrained_model_archive[pretrained_model_name_or_path]
            model_file = load_from_cache(pretrained_model_name_or_path + '.ckpt',
                                         model_url,
                                         force_download=force_download)
        elif pretrained_model_name_or_path in cls.pytorch_pretrained_model_archive_list:
            logging.info("The checkpoint file not found in archive list, "
                         "start to download from torch.")
            model_url = HUGGINGFACE_BASE_URL.format(pretrained_model_name_or_path)
            torch_model_file = load_from_cache(pretrained_model_name_or_path + '.bin',
                                               model_url,
                                               force_download=force_download)
            model_file = convert_state_dict(torch_model_file, cls.name)

        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        # load ckpt
        try:
            param_dict = load_checkpoint(model_file)
        except Exception as exc:
            raise ValueError(f"File {model_file} is not a checkpoint file, "
                             f"please check the path.") from exc

        param_not_load = load_param_into_net(model, param_dict)
        if len(param_not_load) == len(model.trainable_params()):
            raise KeyError(f"The following weights in model are not found: {param_not_load}")

        return model

    def save(self, save_dir: Union[str, os.PathLike]):
        """save pretrained cell"""
        raise NotImplementedError
