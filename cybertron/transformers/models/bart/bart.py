# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpore BART model, ported from Transformers."""
import logging
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

from cybertron.abc import PretrainedCell
from cybertron.common.layers import activation_map, Dense, Embedding
from cybertron.common.ops import dropout, linear, triu
from cybertron.configs.bart import BartConfig

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
    # See all BART models at https://huggingface.co/models?filter=bart
]

def torch_to_mindspore(state_dict):
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
        if 'layer_norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embeddings' in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    return ms_ckpt


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.ndim == 2
    return attention_mask.eq(0)


def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=mindspore.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.shape
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = triu(fill_with_neg_inf(ops.zeros((tgt_len, tgt_len))), 1).astype(causal_mask_dtype)
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PretrainedCell):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Dense):
            module.weight.set_data(initializer(Normal(std), module.weight.shape))
            if module.bias is not None:
                module.bias.set_data(initializer('zeros', module.bias.shape))
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            init_tensor = initializer(Normal(std), module.embedding_table.shape)
            if module.padding_idx is not None:
                init_tensor[module.padding_idx] = 0
            module.embedding_table.set_data(init_tensor)

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = mindspore.Tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], mindspore.int32)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.embedding_table.shape
    lin_layer = Dense(emb_size, vocab_size, has_bias=False)
    lin_layer.weight.set_data(emb.embedding_table.data)
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.copy()
    index_of_eos = (input_ids.ne(pad_token_id).sum(axis=1) - 1).expand_dims(-1)
    prev_output_tokens[:, 0] = input_ids.gather_elements(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.equal(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Cell):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.dropout = config.dropout
        self.activation_fn = activation_map.get(config.activation_function, nn.GELU(False))
        self.activation_dropout = config.activation_dropout
        self.fc1 = Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )

        x = dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


class BartEncoder(nn.Cell):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_size
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.CellList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([embed_dim]) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = nn.LayerNorm([config.d_model]) if config.normalize_before else None

    def construct(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.swapaxes(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.swapaxes(0, 1) for hidden_state in encoder_states]
        x = x.swapaxes(0, 1)

        return x, encoder_states, all_attentions


class DecoderLayer(nn.Cell):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = activation_map.get(config.activation_function, nn.GELU(False))
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.fc1 = Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Cell):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.CellList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = nn.LayerNorm([config.d_model]) if config.normalize_embedding else nn.Identity()
        self.layer_norm = nn.LayerNorm([config.d_model]) if config.add_final_layer_norm else None

    def construct(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.swapaxes(0, 1)
        encoder_hidden_states = encoder_hidden_states.swapaxes(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.swapaxes(0, 1) for hidden_state in all_hidden_states]
        x = x.swapaxes(0, 1)
        encoder_hidden_states = encoder_hidden_states.swapaxes(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class SelfAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = Dense(embed_dim, embed_dim, has_bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.view(dim_0, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)

    def construct(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.shape[1]
        attn_weights = ops.bmm(q, k.swapaxes(1, 2))
        assert attn_weights.shape == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.ndim == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.shape[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.expand_dims(1).expand_dims(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_probs = dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = ops.bmm(attn_probs, v)
        assert attn_output.shape == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(0, 1).view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = ops.cat([prev_key, k], axis=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = ops.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.shape[1], static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = ops.cat([prev_key_padding_mask, key_padding_mask], axis=1)
                return new_key_padding_mask
        else:
            new_key_padding_mask = prev_key_padding_mask
            return new_key_padding_mask

        if key_padding_mask is not None:
            filler = ops.zeros((batch_size, src_len - key_padding_mask.shape[1]), dtype=key_padding_mask.dtype)
            new_key_padding_mask = ops.cat([filler, key_padding_mask], axis=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class BartClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = Dense(input_dim, inner_dim)
        self.dropout = nn.Dropout(1-pooler_dropout)
        self.out_proj = Dense(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def construct(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.new_ones((1, 1)) * (seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = ops.arange(seq_len, dtype=mindspore.int64)
        return super().construct(positions + self.offset)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill(float("-inf")).astype(t.dtype)


def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    outputs = ()
    for x in tup:
        if isinstance(x, mindspore.Tensor) or x:
            outputs += (x,)
    return outputs


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)


class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx=padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def construct(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.embedding_table.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], mindspore.Tensor)
        encoder_outputs = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class BartForConditionalGeneration(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        self.final_logits_bias = Parameter(Tensor(np.zeros((1, self.model.shared.num_embeddings))), 'final_logits_bias')

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = ops.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Conditional generation example::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."

            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]

            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)

            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits = linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if labels is not None:
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = Tensor([x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=mindspore.int64)
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly


class BartForSequenceClassification(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification loss (cross entropy)
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.
        """
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.equal(self.config.eos_token_id)
        if len(ops.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = ops.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BartForQuestionAnswering(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = Dense(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[1:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # return outputs  # (loss), start_logits, end_logits, encoder_outputs, (hidden_states), (attentions)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    def _init_weight(out: Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2].set_data(Tensor(np.sin(position_enc[:, 0::2])))  # This line breaks for odd n_pos
        out[:, dim // 2 :].set_data(Tensor(np.cos(position_enc[:, 1::2])))
        out.requires_grad = False
        return out

    def construct(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.new_ones((1, 1)) * (seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = ops.arange(seq_len, dtype=mindspore.int64)
        return super().construct(positions)
