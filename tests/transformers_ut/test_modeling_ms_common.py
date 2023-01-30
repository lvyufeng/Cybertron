# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import copy
import gc
import inspect
import json
import os
import os.path
import pickle
import random
import sys
import tempfile
import unittest
import unittest.mock as mock
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# import transformers
# from requests.exceptions import HTTPError
# from transformers import (
#     AutoConfig,
#     AutoModel,
#     AutoModelForSequenceClassification,
#     PretrainedConfig,
#     is_torch_available,
#     logging,
# )
from cybertron.transformers.models.auto import get_values
from cybertron.transformers.testing_utils import (
    TOKEN,
    USER,
#     CaptureLogger,
#     TestCasePlus,
    is_flaky,
#     is_pt_flax_cross_test,
#     is_pt_tf_cross_test,
#     is_staging_test,
#     require_accelerate,
#     require_safetensors,
    require_ms,
    is_ms_available,
#     require_torch_gpu,
#     require_torch_multi_gpu,
#     require_usr_bin_time,
    slow,
#     torch_device,
)
# from transformers.utils import (
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
#     is_accelerate_available,
#     is_flax_available,
#     is_tf_available,
#     is_torch_fx_available,
# )
# from transformers.utils.generic import ModelOutput


# sys.path.append(str(Path(__file__).parent.parent / "utils"))

# from test_module.custom_configuration import CustomConfig, NoSuperInitConfig  # noqa E402


# if is_accelerate_available():
#     from accelerate.utils import compute_module_sizes


if is_ms_available():
    import mindspore
    from mindspore import nn, ops

# if is_torch_available():
#     import torch
#     from torch import nn

#     from test_module.custom_modeling import CustomModel, NoSuperInitModel
    from cybertron.transformers import (
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
        MODEL_FOR_AUDIO_XVECTOR_MAPPING,
        MODEL_FOR_BACKBONE_MAPPING,
        MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        # AdaptiveEmbedding,
        AutoModelForCausalLM,
        # AutoTokenizer,
        BertConfig,
        BertModel,
        # PreTrainedModel,
        # T5Config,
        # T5ForConditionalGeneration,
    )
#     from transformers.modeling_utils import shard_checkpoint

#     # Fake pretrained models for tests
#     class BaseModel(PreTrainedModel):
#         config_class = PretrainedConfig

#         def __init__(self, config):
#             super().__init__(config)
#             self.linear = nn.Linear(4, 5)
#             self.linear_2 = nn.Linear(5, 6)

#         def forward(self, x):
#             return self.linear_2(self.linear(x))

#     class ModelWithHead(PreTrainedModel):
#         base_model_prefix = "base"
#         config_class = PretrainedConfig

#         def _init_weights(self, module):
#             pass

#         def __init__(self, config):
#             super().__init__(config)
#             self.base = BaseModel(config)
#             # linear is a common name between Base and Head on purpose.
#             self.linear = nn.Linear(6, 3)
#             self.linear2 = nn.Linear(3, 5)

#         def forward(self, x):
#             return self.linear2(self.linear(self.base(x)))


# if is_tf_available():
#     import tensorflow as tf

# if is_flax_available():
#     import jax.numpy as jnp
#     from transformers.modeling_flax_pytorch_utils import (
#         convert_pytorch_state_dict_to_flax,
#         load_flax_weights_in_pytorch_model,
#     )

# if is_torch_fx_available():
#     from transformers.utils.fx import symbolic_trace


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


# TINY_T5 = "patrickvonplaten/t5-tiny-random"
# TINY_BERT_FOR_TOKEN_CLASSIFICATION = "hf-internal-testing/tiny-bert-for-token-classification"


@require_ms
class ModelTesterMixin:

    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    fx_compatible = False
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_head_masking = True
    test_mismatched_shapes = True
    test_missing_keys = True
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = True
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: v.expand_dims(1).expand(mindspore.Tensor((-1, self.model_tester.num_choices, -1)))
                if isinstance(v, mindspore.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }
        elif model_class in get_values(MODEL_FOR_AUDIO_XVECTOR_MAPPING):
            inputs_dict.pop("attention_mask")

        if return_labels:
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = ops.ones(self.model_tester.batch_size, dtype=mindspore.int32)
            elif model_class in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING),
            ]:
                inputs_dict["start_positions"] = ops.zeros(
                    self.model_tester.batch_size, dtype=mindspore.int32)
                inputs_dict["end_positions"] = ops.zeros(
                    self.model_tester.batch_size, dtype=mindspore.int32)
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = ops.zeros(
                    self.model_tester.batch_size, dtype=mindspore.int32)
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
            ]:
                inputs_dict["labels"] = ops.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=mindspore.int32)
            elif model_class in get_values(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict["bool_masked_pos"] = ops.zeros(
                    (self.model_tester.batch_size, num_patches**2), dtype=mindspore.int32)
            elif model_class in get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING):
                batch_size, num_channels, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = ops.zeros(
                    [self.model_tester.batch_size, height, width], dtype=mindspore.int32)

        return inputs_dict

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.asnumpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.asnumpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)
            first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_save_load_keys_to_ignore_on_save(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            _keys_to_ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
            if _keys_to_ignore_on_save is None:
                continue

            # check the keys are in the original state_dict
            for k in _keys_to_ignore_on_save:
                self.assertIn(k, model.state_dict().keys(), "\n".join(model.state_dict().keys()))

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                output_model_file = os.path.join(tmpdirname, WEIGHTS_NAME)
                state_dict_saved = torch.load(output_model_file)
                for k in _keys_to_ignore_on_save:
                    self.assertNotIn(k, state_dict_saved.keys(), "\n".join(state_dict_saved.keys()))

                # Test we can load the state dict in the model, necessary for the checkpointing API in Trainer.
                load_result = model.load_state_dict(state_dict_saved, strict=False)
                self.assertTrue(
                    len(load_result.missing_keys) == 0
                    or set(load_result.missing_keys) == set(model._keys_to_ignore_on_save)
                )
                self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(config)
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model.gradient_checkpointing_enable()
            self.assertTrue(model.is_gradient_checkpointing)

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)

    @is_flaky()
    def test_save_load_fast_init_from_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(model_class):
                pass

            model_class_copy = CopyClass

            # make sure that all keys are expected for test
            model_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            model_class_copy._init_weights = self._mock_init_weights

            model = base_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = model_class_copy.from_pretrained(tmpdirname)
                model_slow_init = model_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_save_load_fast_init_to_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:

            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = self._mock_init_weights

            model = model_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.config.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = base_class_copy.from_pretrained(tmpdirname)
                model_slow_init = base_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    max_diff = torch.max(
                        torch.abs(model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key])
                    ).item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.asnumpy()
            out_2 = second.asnumpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)
            first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_construct_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.construct)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                    if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                    else ["encoder_outputs"]
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class in [
                *get_values(MODEL_MAPPING),
                *get_values(MODEL_FOR_BACKBONE_MAPPING),
            ]:
                continue

            model = model_class(config)
            model.set_train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            grad_fn = mindspore.value_and_grad(model, None, model.trainable_params())
            outputs, grads = grad_fn(**inputs)
            loss = outputs['loss']

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            if (
                model_class in [*get_values(MODEL_MAPPING), *get_values(MODEL_FOR_BACKBONE_MAPPING)]
                or not model_class.supports_gradient_checkpointing
            ):
                continue
            model = model_class(config)
            model.gradient_checkpointing_enable()
            model.set_train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class in [
                    *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                    *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING),
                ]:
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    # This is copied from `torch/testing/_internal/jit_utils.py::clear_class_registry`
    def clear_torch_jit_class_registry(self):

        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        # torch 1.8 has no `_clear_class_state` in `torch.jit._state`
        if hasattr(torch.jit._state, "_clear_class_state"):
            torch.jit._state._clear_class_state()

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.set_train(False)

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = ops.ones(
                (self.model_tester.num_hidden_layers,
                self.model_tester.num_attention_heads)
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.construct)
                arg_names = [*signature.parameters.keys()]
                if "decoder_head_mask" in arg_names:  # necessary diferentiation because of T5 model
                    inputs["decoder_head_mask"] = head_mask
                if "cross_attn_head_mask" in arg_names:
                    inputs["cross_attn_head_mask"] = head_mask
            outputs = model(**inputs, return_dict=True)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            # multihead_outputs = head_mask.grad

            # self.assertIsNotNone(multihead_outputs)
            # self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        ops.sum(ops.isnan(t).astype(mindspore.int32)), t.numel() / 4
                    )  # Check we don't have more than 25% nans (arbitrary)
                attentions = [
                    t.masked_fill(ops.isnan(t), 0.0) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().asnumpy().item(), 0.0)
                self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().asnumpy().item(), 0.0)
                if len(attentions) > 2:  # encoder-decoder models have only 2 layers in each module
                    self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().asnumpy().item(), 0.0)
                self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().asnumpy().item(), 0.0)
                self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().asnumpy().item(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_head_pruning(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.set_train(False)
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_pretrained(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.set_train(False)
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_config_init(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.set_train(False)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_integration(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {0: [0], 1: [1, 2]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.set_train()

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            heads_to_prune = {0: [0], 2: [1, 2]}
            model.prune_heads(heads_to_prune)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2], 2: [1, 2]})

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.set_train(False)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        if config.is_encoder_decoder:
            # Seq2Seq models
            encoder_hidden_states = outputs.encoder_hidden_states[0]
            encoder_hidden_states.retain_grad()

            decoder_hidden_states = outputs.decoder_hidden_states[0]
            decoder_hidden_states.retain_grad()

            if self.has_attentions:
                encoder_attentions = outputs.encoder_attentions[0]
                encoder_attentions.retain_grad()

                decoder_attentions = outputs.decoder_attentions[0]
                decoder_attentions.retain_grad()

                cross_attentions = outputs.cross_attentions[0]
                cross_attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(encoder_hidden_states.grad)
            self.assertIsNotNone(decoder_hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(encoder_attentions.grad)
                self.assertIsNotNone(decoder_attentions.grad)
                self.assertIsNotNone(cross_attentions.grad)
        else:
            # Encoder-/Decoder-only models
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

            if self.has_attentions:
                attentions = outputs.attentions[0]
                attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(attentions.grad)

    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            mindspore.set_seed(0)
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.set_train(False)

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            mindspore.set_seed(0)
            config.chunk_size_feed_forward = 1
            model = model_class(config)
            model.set_train(False)

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            self.assertTrue(np.allclose(hidden_states_no_chunk.asnumpy(), hidden_states_with_chunk.asnumpy(), atol=1e-3))

    def test_resize_position_vector_embeddings(self):
        if not self.test_resize_position_embeddings:
            return

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            max_position_embeddings = config.max_position_embeddings

            # Retrieve the embeddings and clone theme
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                encoder_cloned_embeddings = encoder_model_embed.weight.clone()
                decoder_cloned_embeddings = decoder_model_embed.weight.clone()
            else:
                model_embed = model.get_position_embeddings()
                cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the position embeddings with a larger max_position_embeddings increases
            # the model's postion embeddings size
            model.resize_position_embeddings(max_position_embeddings + 10)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings + 10)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] + 10)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] + 10)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the position embeddings with a smaller max_position_embeddings decreases
            # the model's max_position_embeddings
            model.resize_position_embeddings(max_position_embeddings - 5)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings - 5)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] - 5)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] - 5)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 5)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True

            if model.config.is_encoder_decoder:
                for p1, p2 in zip(encoder_cloned_embeddings, encoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
                for p1, p2 in zip(decoder_cloned_embeddings, decoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
            else:
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Embedding, AdaptiveEmbedding))
            model.set_input_embeddings(nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            return
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):

                extra_params = {k: v for k, v in model.parameters_and_names() if not k.startswith(base_model_prefix)}
                # extra_params.update({k: v for k, v in model.named_buffers() if not k.startswith(base_model_prefix)})
                # Some models define this as None
                if model._keys_to_ignore_on_load_missing:
                    for key in model._keys_to_ignore_on_load_missing:
                        extra_params.pop(key, None)

                if not extra_params:
                    # In that case, we *are* on a head model, but every
                    # single key is not actual parameters and this is
                    # tested in `test_tied_model_weights_key_ignore` test.
                    continue

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    self.assertGreater(len(loading_info["missing_keys"]), 0, model.__class__.__name__)

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    def test_tied_model_weights_key_ignore(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model_tied = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model_tied.save_pretrained(d)

                # We are nuking ALL weights on file, so every parameter should
                # yell on load. We're going to detect if we yell too much, or too little.
                with open(os.path.join(d, "pytorch_model.bin"), "wb") as f:
                    torch.save({}, f)
                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)

                # ! Actually we could use `state_dict()` and check iteratively the tensors which are the same (for instance using `tensor.data_ptr()`). to detect the duplicates.
                # ```python
                # model = GPT2LMHeadModel.from_pretrained("gpt2")
                # "lm_head.weight" in model.state_dict().keys()  # True
                # "lm_head.weight" in model.named_parameters() # False
                # In [6]: model.lm_head.weight.data_ptr()
                # Out[6]: 139901378371648
                # In [9]: model.transformer.wte.weight.data_ptr()
                # Out[9]: 139901378371648  # Same PTR, it's the same DATA ! we would need to check for stride too to be 100% accurate.
                # ```

                prefix = f"{model_reloaded.base_model_prefix}."
                params = dict(model_reloaded.named_parameters())
                params.update(dict(model_reloaded.named_buffers()))
                # param_names = set(k[len(prefix) :] if k.startswith(prefix) else k for k in params.keys())
                param_names = set(k[len(prefix) :] if k.startswith(prefix) else k for k in params.keys())

                missing_keys = set(infos["missing_keys"])

                extra_missing = missing_keys - param_names
                # missed_missing = param_names - missing_keys

                self.assertEqual(
                    extra_missing,
                    set(),
                    f"This model {model_class.__name__} might be missing some `keys_to_ignore`: {extra_missing}",
                )

                # self.assertEqual(
                #     missed_missing,
                #     set(),
                #     f"This model {model_class.__name__} ignores keys {missed_missing} but they look like real"
                #     " parameters",
                # )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _make_attention_mask_non_null(self, inputs_dict):
        """Make sure no sequence has all zeros as attention mask"""

        for k in ["attention_mask", "encoder_attention_mask", "decoder_attention_mask"]:
            if k in inputs_dict:
                attention_mask = inputs_dict[k]

                # Make sure no all 0s attention masks - to avoid failure at this moment.
                # Put `1` at the beginning of sequences to make it still work when combining causal attention masks.
                # TODO: remove this line once a fix regarding large negative values for attention mask is done.
                attention_mask = torch.cat(
                    [torch.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], dim=-1
                )

                # Here we make the first sequence with all 0s as attention mask.
                # Currently, this will fail for `TFWav2Vec2Model`. This is caused by the different large negative
                # values, like `1e-4`, `1e-9`, `1e-30` and `-inf` for attention mask across models/frameworks.
                # TODO: enable this block once the large negative values thing is cleaned up.
                # (see https://github.com/huggingface/transformers/issues/14859)
                # attention_mask = torch.cat(
                #     [torch.zeros_like(attention_mask[:1], dtype=attention_mask.dtype), attention_mask[1:]],
                #     dim=0
                # )

                inputs_dict[k] = attention_mask

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _postprocessing_to_ignore_test_cases(self, tf_outputs, pt_outputs, model_class):
        """For temporarily ignoring some failed test cases (issues to be fixed)"""

        tf_keys = set([k for k, v in tf_outputs.items() if v is not None])
        pt_keys = set([k for k, v in pt_outputs.items() if v is not None])

        key_differences = tf_keys.symmetric_difference(pt_keys)

        if model_class.__name__ in [
            "FlaubertWithLMHeadModel",
            "FunnelForPreTraining",
            "ElectraForPreTraining",
            "XLMWithLMHeadModel",
            "TransfoXLLMHeadModel",
        ]:
            for k in key_differences:
                if k in ["loss", "losses"]:
                    tf_keys.discard(k)
                    pt_keys.discard(k)
        elif model_class.__name__.startswith("GPT2"):
            # `TFGPT2` has `past_key_values` as a tensor while `GPT2` has it as a tuple.
            tf_keys.discard("past_key_values")
            pt_keys.discard("past_key_values")

        # create new outputs from the remaining fields
        new_tf_outputs = type(tf_outputs)(**{k: tf_outputs[k] for k in tf_keys})
        new_pt_outputs = type(pt_outputs)(**{k: pt_outputs[k] for k in pt_keys})

        return new_tf_outputs, new_pt_outputs

    # Copied from tests.test_modeling_tf_common.TFModelTesterMixin.check_pt_tf_outputs
    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=1e-5, name="outputs", attributes=None):
        """Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.

        Args:
            model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative
                error messages.
            name (`str`): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.
            attributes (`Tuple[str]`): The names of the output's element if the output is a tuple/list with each element
                being a named field in the output.
        """

        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(tf_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is",
            )

            # Don't copy this block to model specific test file!
            # TODO: remove this method and this line after issues are fixed
            tf_outputs, pt_outputs = self._postprocessing_to_ignore_test_cases(tf_outputs, pt_outputs, model_class)

            tf_keys = [k for k, v in tf_outputs.items() if v is not None]
            pt_keys = [k for k, v in pt_outputs.items() if v is not None]

            self.assertEqual(tf_keys, pt_keys, f"{name}: Output keys differ between TF and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `name`
            attributes = tuple([f"{name}.{k}" for k in tf_keys])
            self.check_pt_tf_outputs(
                tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(tf_outputs) in [tuple, list]:
            self.assertEqual(type(tf_outputs), type(pt_outputs), f"{name}: Output types differ between TF and PyTorch")
            self.assertEqual(len(tf_outputs), len(pt_outputs), f"{name}: Output lengths differ between TF and PyTorch")

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(tf_outputs),
                    f"{name}: The tuple `attributes` should have the same length as `tf_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `name`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(tf_outputs))])

            for tf_output, pt_output, attr in zip(tf_outputs, pt_outputs, attributes):
                self.check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(tf_outputs, tf.Tensor):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `tf_outputs` is"
            )

            tf_outputs = tf_outputs.numpy()
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                tf_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between TF and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(tf_outputs):
                tf_outputs = np.array([tf_outputs])
                pt_outputs = np.array([pt_outputs])

            tf_nans = np.isnan(tf_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[tf_nans] = 0
            tf_outputs[tf_nans] = 0
            pt_outputs[pt_nans] = 0
            tf_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
            self.assertLessEqual(max_diff, tol, f"{name}: Difference between PyTorch and TF is {max_diff} (>= {tol}).")
        else:
            raise ValueError(
                "`tf_outputs` should be an instance of `ModelOutput`, a `tuple`, or an instance of `tf.Tensor`. Got"
                f" {type(tf_outputs)} instead."
            )

    def prepare_tf_inputs_from_pt_inputs(self, pt_inputs_dict):

        tf_inputs_dict = {}
        for key, tensor in pt_inputs_dict.items():
            # skip key that does not exist in tf
            if type(tensor) == bool:
                tf_inputs_dict[key] = tensor
            elif key == "input_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            elif key == "pixel_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            elif key == "input_features":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            # other general float inputs
            elif tensor.is_floating_point():
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            else:
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.int32)

        return tf_inputs_dict

    def check_pt_tf_models(self, tf_model, pt_model, pt_inputs_dict):

        tf_inputs_dict = self.prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)

        # send pytorch inputs to the correct device
        pt_inputs_dict = {
            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs_dict.items()
        }

        # send pytorch model to the correct device
        pt_model.to(torch_device)

        # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
        pt_model.eval()

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_dict)
        tf_outputs = tf_model(tf_inputs_dict)

        # tf models returned loss is usually a tensor rather than a scalar.
        # (see `hf_compute_loss`: it uses `tf.keras.losses.Reduction.NONE`)
        # Change it here to a scalar to match PyTorch models' loss
        tf_loss = getattr(tf_outputs, "loss", None)
        if tf_loss is not None:
            tf_outputs.loss = tf.math.reduce_mean(tf_loss)

        self.check_pt_tf_outputs(tf_outputs, pt_outputs, type(pt_model))

    # @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self):
        import transformers

        for model_class in self.all_model_classes:

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            tf_model_class_name = "TF" + model_class.__name__  # Add the "TF" at the beginning
            if not hasattr(transformers, tf_model_class_name):
                # transformers does not have this model in TF version yet
                return

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            # TODO: Use a uniform value for all models, make sure all tests pass without this processing, and remove it.
            self._make_attention_mask_non_null(inputs_dict)

            tf_model_class = getattr(transformers, tf_model_class_name)

            pt_model = model_class(config)
            tf_model = tf_model_class(config)

            pt_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            pt_inputs_dict_with_labels = self._prepare_for_class(
                inputs_dict,
                model_class,
                # Not all models accept "labels" in the forward pass (yet :) )
                return_labels=True if "labels" in inspect.signature(model_class.forward).parameters.keys() else False,
            )

            # make sure only tf inputs are forward that actually exist in function args
            tf_input_keys = set(inspect.signature(tf_model.call).parameters.keys())

            # remove all head masks
            tf_input_keys.discard("head_mask")
            tf_input_keys.discard("cross_attn_head_mask")
            tf_input_keys.discard("decoder_head_mask")

            pt_inputs_dict = {k: v for k, v in pt_inputs_dict.items() if k in tf_input_keys}
            pt_inputs_dict_with_labels = {k: v for k, v in pt_inputs_dict_with_labels.items() if k in tf_input_keys}

            # For some models (e.g. base models), there is no label returned.
            # Set the input dict to `None` to avoid check outputs twice for the same input dicts.
            if set(pt_inputs_dict_with_labels.keys()).symmetric_difference(pt_inputs_dict.keys()):
                pt_inputs_dict_with_labels = None

            # Check we can load pt model in tf and vice-versa with model => model functions
            # Here requires `tf_inputs_dict` to build `tf_model`
            tf_inputs_dict = self.prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)
            tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=tf_inputs_dict)
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)
            # check with `labels`
            if pt_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict_with_labels)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)
            # check with `labels`
            if pt_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict_with_labels)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=1e-5, name="outputs", attributes=None):
        """
        Args:
            model_class: The class of the model that is currently testing. For example, ..., etc.
            Currently unused, but it could make debugging easier and faster.

            names: A string, or a list of strings. These specify what fx_outputs/pt_outputs represent in the model outputs.
                Currently unused, but in the future, we could use this information to make the error message clearer
                by giving the name(s) of the output tensor(s) with large difference(s) between PT and Flax.
        """

        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(fx_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `fx_outputs` is",
            )

            fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
            pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

            self.assertEqual(fx_keys, pt_keys, f"{name}: Output keys differ between Flax and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `name`
            attributes = tuple([f"{name}.{k}" for k in fx_keys])
            self.check_pt_flax_outputs(
                fx_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(fx_outputs) in [tuple, list]:
            self.assertEqual(
                type(fx_outputs), type(pt_outputs), f"{name}: Output types differ between Flax and PyTorch"
            )
            self.assertEqual(
                len(fx_outputs), len(pt_outputs), f"{name}: Output lengths differ between Flax and PyTorch"
            )

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(fx_outputs),
                    f"{name}: The tuple `attributes` should have the same length as `fx_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `name`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(fx_outputs))])

            for fx_output, pt_output, attr in zip(fx_outputs, pt_outputs, attributes):
                self.check_pt_flax_outputs(fx_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(fx_outputs, jnp.ndarray):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `fx_outputs` is"
            )

            # Using `np.asarray` gives `ValueError: assignment destination is read-only` at the line `fx_outputs[fx_nans] = 0`.
            fx_outputs = np.array(fx_outputs)
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                fx_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between Flax and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(fx_outputs):
                fx_outputs = np.array([fx_outputs])
                pt_outputs = np.array([pt_outputs])

            fx_nans = np.isnan(fx_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[fx_nans] = 0
            fx_outputs[fx_nans] = 0
            pt_outputs[pt_nans] = 0
            fx_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(fx_outputs - pt_outputs))
            self.assertLessEqual(
                max_diff, tol, f"{name}: Difference between PyTorch and Flax is {max_diff} (>= {tol})."
            )
        else:
            raise ValueError(
                "`fx_outputs` should be an instance of `ModelOutput`, a `tuple`, or an instance of `jnp.ndarray`. Got"
                f" {type(fx_outputs)} instead."
            )


    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    # @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask", "decoder_head_mask", "cross_attn_head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    # @require_torch_multi_gpu
    def test_model_parallelization(self):
        if not self.test_model_parallel:
            return

        # a candidate for testing_utils
        def get_current_gpu_memory_use():
            """returns a list of cuda memory allocations per GPU in MBs"""

            per_device_memory = []
            for id in range(torch.cuda.device_count()):
                with torch.cuda.device(id):
                    per_device_memory.append(torch.cuda.memory_allocated() >> 20)

            return per_device_memory

        # Needs a large model to see the difference.
        config = self.model_tester.get_large_model_config()

        for model_class in self.all_parallelizable_model_classes:
            torch.cuda.empty_cache()

            # 1. single gpu memory load + unload + memory measurements
            # Retrieve initial memory usage (can easily be ~0.6-1.5GB if cuda-kernels have been preloaded by previous tests)
            memory_at_start = get_current_gpu_memory_use()

            # Put model on device 0 and take a memory snapshot
            model = model_class(config)
            model.to("cuda:0")
            memory_after_model_load = get_current_gpu_memory_use()

            # The memory use on device 0 should be higher than it was initially.
            self.assertGreater(memory_after_model_load[0], memory_at_start[0])

            del model
            gc.collect()
            torch.cuda.empty_cache()

            # 2. MP test
            # it's essential to re-calibrate the usage before the next stage
            memory_at_start = get_current_gpu_memory_use()

            # Spread model layers over multiple devices
            model = model_class(config)
            model.parallelize()
            memory_after_parallelization = get_current_gpu_memory_use()

            # Assert that the memory use on all devices is higher than it was when loaded only on CPU
            for n in range(len(model.device_map.keys())):
                self.assertGreater(memory_after_parallelization[n], memory_at_start[n])

            # Assert that the memory use of device 0 is lower than it was when the entire model was loaded on it
            self.assertLess(memory_after_parallelization[0], memory_after_model_load[0])

            # Assert that the memory use of device 1 is higher than it was when the entire model was loaded
            # on device 0 and device 1 wasn't used at all
            self.assertGreater(memory_after_parallelization[1], memory_after_model_load[1])

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # @require_torch_multi_gpu
    def test_model_parallel_equal_results(self):
        if not self.test_model_parallel:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_parallelizable_model_classes:
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            def cast_to_device(dictionary, device):
                output = {}
                for k, v in dictionary.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.to(device)
                    else:
                        output[k] = v

                return output

            model = model_class(config)
            output = model(**cast_to_device(inputs_dict, "cpu"))

            model.parallelize()

            parallel_output = model(**cast_to_device(inputs_dict, "cuda:0"))

            for value, parallel_value in zip(output, parallel_output):
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.allclose(value, parallel_value.to("cpu"), atol=1e-7))
                elif isinstance(value, (Tuple, List)):
                    for value_, parallel_value_ in zip(value, parallel_value):
                        self.assertTrue(torch.allclose(value_, parallel_value_.to("cpu"), atol=1e-7))

    # @require_torch_multi_gpu
    def test_model_parallel_beam_search(self):
        if not self.test_model_parallel:
            return

        all_generative_and_parallelizable_model_classes = tuple(
            set(self.all_generative_model_classes).intersection(self.all_parallelizable_model_classes)
        )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in all_generative_and_parallelizable_model_classes:
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)

            def cast_to_device(dictionary, device):
                output = {}
                for k, v in dictionary.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.to(device)
                    else:
                        output[k] = v

                return output

            model.parallelize()
            model.generate(**cast_to_device(inputs_dict, "cuda:0"), num_beams=2)

    def check_device_map_is_respected(self, model, device_map):
        for param_name, param in model.named_parameters():
            # Find device in device_map
            while len(param_name) > 0 and param_name not in device_map:
                param_name = ".".join(param_name.split(".")[:-1])
            if param_name not in device_map:
                raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

            param_device = device_map[param_name]
            if param_device in ["cpu", "disk"]:
                self.assertEqual(param.device, torch.device("meta"))
            else:
                self.assertEqual(param.device, torch.device(param_device))

    # @require_accelerate
    # @require_torch_gpu
    def test_disk_offload(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            max_size = int(self.model_split_percents[0] * model_size)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                max_memory = {0: max_size, "cpu": max_size}
                with self.assertRaises(ValueError):
                    # This errors out cause it's missing an offload folder
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

                new_model = model_class.from_pretrained(
                    tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                self.assertTrue(torch.allclose(base_output[0], new_output[0]))

    # @require_accelerate
    # @require_torch_gpu
    def test_cpu_offload(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    self.assertTrue(torch.allclose(base_output[0], new_output[0]))

    # @require_accelerate
    # @require_torch_multi_gpu
    def test_model_parallelism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})

                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    self.assertTrue(torch.allclose(base_output[0], new_output[0]))

    def test_problem_types(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        problem_types = [
            {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
            {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
            {"title": "regression", "num_labels": 1, "dtype": torch.float},
        ]

        for model_class in self.all_model_classes:
            if model_class not in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                continue

            for problem_type in problem_types:
                with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):

                    config.problem_type = problem_type["title"]
                    config.num_labels = problem_type["num_labels"]

                    model = model_class(config)
                    model.to(torch_device)
                    model.train()

                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                    if problem_type["num_labels"] > 1:
                        inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                    inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                    # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                    # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                    # they have the same size." which is a symptom something in wrong for the regression problem.
                    # See https://github.com/huggingface/transformers/issues/11780
                    with warnings.catch_warnings(record=True) as warning_list:
                        loss = model(**inputs).loss
                    for w in warning_list:
                        if "Using a target size that is different to the input size" in str(w.message):
                            raise ValueError(
                                f"Something is going wrong in the regression problem: intercepted {w.message}"
                            )

                    loss.backward()

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(RuntimeError):
                        new_model_without_prefix = AutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return mindspore.Tensor(values, dtype=mindspore.int32).view(shape)


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return mindspore.Tensor(values, dtype=mindspore.float32).view(shape)


# def check_models_equal(model1, model2):
#     models_are_equal = True
#     for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
#         if model1_p.data.ne(model2_p.data).sum() > 0:
#             models_are_equal = False

#     return models_are_equal


# @require_torch
# class ModelUtilsTest(TestCasePlus):
#     @slow
#     def test_model_from_pretrained(self):
#         for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
#             config = BertConfig.from_pretrained(model_name)
#             self.assertIsNotNone(config)
#             self.assertIsInstance(config, PretrainedConfig)

#             model = BertModel.from_pretrained(model_name)
#             model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
#             self.assertIsNotNone(model)
#             self.assertIsInstance(model, PreTrainedModel)

#             self.assertEqual(len(loading_info["missing_keys"]), 0)
#             self.assertEqual(len(loading_info["unexpected_keys"]), 8)
#             self.assertEqual(len(loading_info["mismatched_keys"]), 0)
#             self.assertEqual(len(loading_info["error_msgs"]), 0)

#             config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

#             # Not sure this is the intended behavior. TODO fix Lysandre & Thom
#             config.name_or_path = model_name

#             model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
#             self.assertEqual(model.config.output_hidden_states, True)
#             self.assertEqual(model.config, config)

#     def test_model_from_pretrained_subfolder(self):
#         config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
#         model = BertModel(config)

#         subfolder = "bert"
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(os.path.join(tmp_dir, subfolder))

#             with self.assertRaises(OSError):
#                 _ = BertModel.from_pretrained(tmp_dir)

#             model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

#         self.assertTrue(check_models_equal(model, model_loaded))

#     def test_model_from_pretrained_subfolder_sharded(self):
#         config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
#         model = BertModel(config)

#         subfolder = "bert"
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size="10KB")

#             with self.assertRaises(OSError):
#                 _ = BertModel.from_pretrained(tmp_dir)

#             model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

#         self.assertTrue(check_models_equal(model, model_loaded))

#     def test_model_from_pretrained_hub_subfolder(self):
#         subfolder = "bert"
#         model_id = "hf-internal-testing/tiny-random-bert-subfolder"
#         with self.assertRaises(OSError):
#             _ = BertModel.from_pretrained(model_id)

#         model = BertModel.from_pretrained(model_id, subfolder=subfolder)

#         self.assertIsNotNone(model)

#     def test_model_from_pretrained_hub_subfolder_sharded(self):
#         subfolder = "bert"
#         model_id = "hf-internal-testing/tiny-random-bert-sharded-subfolder"
#         with self.assertRaises(OSError):
#             _ = BertModel.from_pretrained(model_id)

#         model = BertModel.from_pretrained(model_id, subfolder=subfolder)

#         self.assertIsNotNone(model)

#     def test_model_from_pretrained_with_different_pretrained_model_name(self):
#         model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
#         self.assertIsNotNone(model)

#         logger = logging.get_logger("transformers.configuration_utils")
#         with CaptureLogger(logger) as cl:
#             BertModel.from_pretrained(TINY_T5)
#         self.assertTrue("You are using a model of type t5 to instantiate a model of type bert" in cl.out)

#     @require_torch
#     def test_model_from_config_torch_dtype(self):
#         # test that the model can be instantiated with dtype of user's choice - as long as it's a
#         # float dtype. To make it happen config.torch_dtype needs to be set before instantiating the
#         # model from the config object.

#         config = T5Config.from_pretrained(TINY_T5)
#         model = AutoModel.from_config(config)
#         # XXX: isn't supported
#         # model = T5ForConditionalGeneration.from_config(config)
#         self.assertEqual(model.dtype, torch.float32)

#         model = AutoModel.from_config(config, torch_dtype=torch.float16)
#         self.assertEqual(model.dtype, torch.float16)

#         # torch.set_default_dtype() supports only float dtypes, so will fail with non-float type
#         with self.assertRaises(ValueError):
#             model = AutoModel.from_config(config, torch_dtype=torch.int64)

#     @require_torch
#     def test_model_from_pretrained_torch_dtype(self):
#         # test that the model can be instantiated with dtype of either
#         # 1. explicit from_pretrained's torch_dtype argument
#         # 2. via autodiscovery by looking at model weights (torch_dtype="auto")
#         # so if a model.half() was saved, we want it to be instantiated as such.
#         #
#         # test an explicit model class, but also AutoModel separately as the latter goes through a different code path
#         model_path = self.get_auto_remove_tmp_dir()

#         # baseline - we know TINY_T5 is fp32 model
#         model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
#         self.assertEqual(model.dtype, torch.float32)

#         # test the default fp32 save_pretrained => from_pretrained cycle
#         model.save_pretrained(model_path)
#         model = T5ForConditionalGeneration.from_pretrained(model_path)
#         self.assertEqual(model.dtype, torch.float32)
#         # test with auto-detection
#         model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
#         self.assertEqual(model.dtype, torch.float32)

#         # test forced loading in fp16 (even though the weights are in fp32)
#         model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
#         self.assertEqual(model.dtype, torch.float16)

#         # test fp16 save_pretrained, loaded with auto-detection
#         model = model.half()
#         model.save_pretrained(model_path)
#         model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
#         self.assertEqual(model.config.torch_dtype, torch.float16)
#         self.assertEqual(model.dtype, torch.float16)

#         # tests `config.torch_dtype` saving
#         with open(f"{model_path}/config.json") as f:
#             config_dict = json.load(f)
#         self.assertEqual(config_dict["torch_dtype"], "float16")

#         # test fp16 save_pretrained, loaded with the explicit fp16
#         model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
#         self.assertEqual(model.dtype, torch.float16)

#         # test AutoModel separately as it goes through a different path
#         # test auto-detection
#         model = AutoModel.from_pretrained(TINY_T5, torch_dtype="auto")
#         self.assertEqual(model.dtype, torch.float32)
#         # test forcing an explicit dtype
#         model = AutoModel.from_pretrained(TINY_T5, torch_dtype=torch.float16)
#         self.assertEqual(model.dtype, torch.float16)

#         # test model whose first param is not of a floating type, but int
#         model = AutoModel.from_pretrained(TINY_BERT_FOR_TOKEN_CLASSIFICATION, torch_dtype="auto")
#         self.assertEqual(model.dtype, torch.float32)

#     def test_no_super_init_config_and_model(self):
#         config = NoSuperInitConfig(attribute=32)
#         model = NoSuperInitModel(config)

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(tmp_dir)

#             new_model = NoSuperInitModel.from_pretrained(tmp_dir)

#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#     def test_shard_checkpoint(self):
#         # This is the model we will use, total size 340,000 bytes.
#         model = torch.nn.Sequential(
#             torch.nn.Linear(100, 200, bias=False),  # size 80,000
#             torch.nn.Linear(200, 200, bias=False),  # size 160,000
#             torch.nn.Linear(200, 100, bias=False),  # size 80,000
#             torch.nn.Linear(100, 50, bias=False),  # size 20,000
#         )
#         state_dict = model.state_dict()

#         with self.subTest("No shard when max size is bigger than model size"):
#             shards, index = shard_checkpoint(state_dict)
#             self.assertIsNone(index)
#             self.assertDictEqual(shards, {WEIGHTS_NAME: state_dict})

#         with self.subTest("Test sharding, no weights bigger than max size"):
#             shards, index = shard_checkpoint(state_dict, max_shard_size="300kB")
#             # Split is first two layers then last two.
#             self.assertDictEqual(
#                 index,
#                 {
#                     "metadata": {"total_size": 340000},
#                     "weight_map": {
#                         "0.weight": "pytorch_model-00001-of-00002.bin",
#                         "1.weight": "pytorch_model-00001-of-00002.bin",
#                         "2.weight": "pytorch_model-00002-of-00002.bin",
#                         "3.weight": "pytorch_model-00002-of-00002.bin",
#                     },
#                 },
#             )

#             shard1 = {"0.weight": state_dict["0.weight"], "1.weight": state_dict["1.weight"]}
#             shard2 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
#             self.assertDictEqual(
#                 shards, {"pytorch_model-00001-of-00002.bin": shard1, "pytorch_model-00002-of-00002.bin": shard2}
#             )

#         with self.subTest("Test sharding with weights bigger than max size"):
#             shards, index = shard_checkpoint(state_dict, max_shard_size="100kB")
#             # Split is first layer, second layer then last 2.
#             self.assertDictEqual(
#                 index,
#                 {
#                     "metadata": {"total_size": 340000},
#                     "weight_map": {
#                         "0.weight": "pytorch_model-00001-of-00003.bin",
#                         "1.weight": "pytorch_model-00002-of-00003.bin",
#                         "2.weight": "pytorch_model-00003-of-00003.bin",
#                         "3.weight": "pytorch_model-00003-of-00003.bin",
#                     },
#                 },
#             )

#             shard1 = {"0.weight": state_dict["0.weight"]}
#             shard2 = {"1.weight": state_dict["1.weight"]}
#             shard3 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
#             self.assertDictEqual(
#                 shards,
#                 {
#                     "pytorch_model-00001-of-00003.bin": shard1,
#                     "pytorch_model-00002-of-00003.bin": shard2,
#                     "pytorch_model-00003-of-00003.bin": shard3,
#                 },
#             )

#     def test_checkpoint_sharding_local(self):
#         model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
#             for max_size in ["50kB", "50kiB", "100kB", "100kiB", "200kB", "200kiB"]:
#                 model.save_pretrained(tmp_dir, max_shard_size=max_size)

#                 # Get each shard file and its size
#                 shard_to_size = {}
#                 for shard in os.listdir(tmp_dir):
#                     if shard.endswith(".bin"):
#                         shard_file = os.path.join(tmp_dir, shard)
#                         shard_to_size[shard_file] = os.path.getsize(shard_file)

#                 index_file = os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)
#                 # Check there is an index but no regular weight file
#                 self.assertTrue(os.path.isfile(index_file))
#                 self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

#                 # Check a file is bigger than max_size only when it has a single weight
#                 for shard_file, size in shard_to_size.items():
#                     if max_size.endswith("kiB"):
#                         max_size_int = int(max_size[:-3]) * 2**10
#                     else:
#                         max_size_int = int(max_size[:-2]) * 10**3
#                     # Note: pickle adds some junk so the weight of the file can end up being slightly bigger than
#                     # the size asked for (since we count parameters)
#                     if size >= max_size_int + 50000:
#                         state_dict = torch.load(shard_file)
#                         self.assertEqual(len(state_dict), 1)

#                 # Check the index and the shard files found match
#                 with open(index_file, "r", encoding="utf-8") as f:
#                     index = json.loads(f.read())

#                 all_shards = set(index["weight_map"].values())
#                 shards_found = set(f for f in os.listdir(tmp_dir) if f.endswith(".bin"))
#                 self.assertSetEqual(all_shards, shards_found)

#                 # Finally, check the model can be reloaded
#                 new_model = BertModel.from_pretrained(tmp_dir)
#                 for p1, p2 in zip(model.parameters(), new_model.parameters()):
#                     self.assertTrue(torch.allclose(p1, p2))

#     def test_checkpoint_sharding_from_hub(self):
#         model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
#         # the model above is the same as the model below, just a sharded version.
#         ref_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
#         for p1, p2 in zip(model.parameters(), ref_model.parameters()):
#             self.assertTrue(torch.allclose(p1, p2))

#     @require_accelerate
#     def test_from_pretrained_low_cpu_mem_usage_functional(self):
#         # test that we can use `from_pretrained(..., low_cpu_mem_usage=True)` with normal and
#         # sharded models

#         mnames = [
#             "hf-internal-testing/tiny-random-bert-sharded",
#             "hf-internal-testing/tiny-random-bert",
#         ]
#         for mname in mnames:
#             _ = BertModel.from_pretrained(mname, low_cpu_mem_usage=True)

#     @require_usr_bin_time
#     @require_accelerate
#     def test_from_pretrained_low_cpu_mem_usage_measured(self):
#         # test that `from_pretrained(..., low_cpu_mem_usage=True)` uses less cpu memory than default

#         mname = "bert-base-cased"

#         preamble = "from transformers import AutoModel"
#         one_liner_str = f'{preamble}; AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=False)'
#         max_rss_normal = self.python_one_liner_max_rss(one_liner_str)
#         # print(f"{max_rss_normal=}")

#         one_liner_str = f'{preamble};  AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=True)'
#         max_rss_low_mem = self.python_one_liner_max_rss(one_liner_str)
#         # print(f"{max_rss_low_mem=}")

#         diff_bytes = max_rss_normal - max_rss_low_mem
#         diff_percent = diff_bytes / max_rss_low_mem
#         # print(f"{diff_bytes=}, {diff_percent=}")
#         # ideally we would compare that the diff is close to ~1x checkpoint size in bytes, but
#         # measuring cpu memory on linux is very tricky and inconsistent, so instead let's check that
#         # it's at least 15% less cpu memory consumed

#         self.assertGreater(
#             diff_percent,
#             0.15,
#             "should use less CPU memory for low_cpu_mem_usage=True, "
#             f"but got max_rss_normal={max_rss_normal} and max_rss_low_mem={max_rss_low_mem}",
#         )

#         # if you want to compare things manually, let's first look at the size of the model in bytes
#         # model = BertModel.from_pretrained(mname, low_cpu_mem_usage=False)
#         # total_numel = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
#         # total_bytes = total_numel * 4  # 420MB
#         # Now the diff_bytes should be very close to total_bytes, but the reports are inconsistent.
#         # The easiest way to test this is to switch the model and torch.load to do all the work on
#         # gpu - that way one can measure exactly the total and peak memory used. Perhaps once we add
#         # functionality to load models directly on gpu, this test can be rewritten to use torch's
#         # cuda memory tracking and then we should be able to do a much more precise test.

#     @require_accelerate
#     @require_torch_multi_gpu
#     @slow
#     def test_model_parallelism_gpt2(self):
#         device_map = {"transformer.wte": 0, "transformer.wpe": 0, "lm_head": 0, "transformer.ln_f": 1}
#         for i in range(12):
#             device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

#         model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=device_map)

#         tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         inputs = tokenizer("Hello, my name is", return_tensors="pt")
#         output = model.generate(inputs["input_ids"].to(0))

#         text_output = tokenizer.decode(output[0].tolist())
#         self.assertEqual(text_output, "Hello, my name is John. I'm a writer, and I'm a writer. I'm")

#     @require_accelerate
#     @require_torch_gpu
#     def test_from_pretrained_disk_offload_task_model(self):
#         model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")
#         device_map = {
#             "transformer.wte": 0,
#             "transformer.wpe": 0,
#             "transformer.h.0": "cpu",
#             "transformer.h.1": "cpu",
#             "transformer.h.2": "cpu",
#             "transformer.h.3": "disk",
#             "transformer.h.4": "disk",
#             "transformer.ln_f": 0,
#             "lm_head": 0,
#         }
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             inputs = torch.tensor([[1, 2, 3]]).to(0)

#             model.save_pretrained(tmp_dir)
#             new_model = AutoModelForCausalLM.from_pretrained(tmp_dir).to(0)
#             outputs1 = new_model.to(0)(inputs)

#             offload_folder = os.path.join(tmp_dir, "offload")
#             new_model_with_offload = AutoModelForCausalLM.from_pretrained(
#                 tmp_dir, device_map=device_map, offload_folder=offload_folder
#             )
#             outputs2 = new_model_with_offload(inputs)

#             self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

#             # With state dict temp offload
#             offload_folder = os.path.join(tmp_dir, "offload")
#             new_model_with_offload = AutoModelForCausalLM.from_pretrained(
#                 tmp_dir,
#                 device_map=device_map,
#                 offload_folder=offload_folder,
#                 offload_state_dict=True,
#             )
#             outputs2 = new_model_with_offload(inputs)

#             self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

#     def test_cached_files_are_used_when_internet_is_down(self):
#         # A mock response for an HTTP head request to emulate server down
#         response_mock = mock.Mock()
#         response_mock.status_code = 500
#         response_mock.headers = {}
#         response_mock.raise_for_status.side_effect = HTTPError
#         response_mock.json.return_value = {}

#         # Download this model to make sure it's in the cache.
#         _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

#         # Under the mock environment we get a 500 error when trying to reach the model.
#         with mock.patch("requests.request", return_value=response_mock) as mock_head:
#             _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
#             # This check we did call the fake head request
#             mock_head.assert_called()

#     def test_load_from_one_file(self):
#         try:
#             tmp_file = tempfile.mktemp()
#             with open(tmp_file, "wb") as f:
#                 http_get(
#                     "https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin", f
#                 )

#             config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
#             _ = BertModel.from_pretrained(tmp_file, config=config)
#         finally:
#             os.remove(tmp_file)

#     def test_legacy_load_from_url(self):
#         # This test is for deprecated behavior and can be removed in v5
#         config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
#         _ = BertModel.from_pretrained(
#             "https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin", config=config
#         )

#     @require_safetensors
#     def test_safetensors_save_and_load(self):
#         model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(tmp_dir, safe_serialization=True)
#             # No pytorch_model.bin file, only a model.safetensors
#             self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
#             self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

#             new_model = BertModel.from_pretrained(tmp_dir)

#             # Check models are equal
#             for p1, p2 in zip(model.parameters(), new_model.parameters()):
#                 self.assertTrue(torch.allclose(p1, p2))

#     @require_safetensors
#     def test_safetensors_load_from_hub(self):
#         safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors")
#         pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

#         # Check models are equal
#         for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
#             self.assertTrue(torch.allclose(p1, p2))

#     @require_safetensors
#     def test_safetensors_save_and_load_sharded(self):
#         model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
#             # No pytorch_model.bin index file, only a model.safetensors index
#             self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))
#             self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
#             # No regular weights file
#             self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
#             self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

#             new_model = BertModel.from_pretrained(tmp_dir)

#             # Check models are equal
#             for p1, p2 in zip(model.parameters(), new_model.parameters()):
#                 self.assertTrue(torch.allclose(p1, p2))

#     @require_safetensors
#     def test_safetensors_load_from_hub_sharded(self):
#         safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded-safetensors")
#         pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")

#         # Check models are equal
#         for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
#             self.assertTrue(torch.allclose(p1, p2))

#     def test_base_model_to_head_model_load(self):
#         base_model = BaseModel(PretrainedConfig())
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             base_model.save_pretrained(tmp_dir)

#             # Can load a base model in a model with head
#             model = ModelWithHead.from_pretrained(tmp_dir)
#             for p1, p2 in zip(model.base.parameters(), base_model.parameters()):
#                 self.assertTrue(torch.allclose(p1, p2))

#             # It doesn't work if the state dict has a mix of keys of the head and base without prefix though.
#             base_state_dict = base_model.state_dict()
#             head_state_dict = model.state_dict()
#             base_state_dict["linear2.weight"] = head_state_dict["linear2.weight"]
#             base_state_dict["linear2.bias"] = head_state_dict["linear2.bias"]
#             torch.save(base_state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))

#             with self.assertRaisesRegex(
#                 ValueError, "The state dictionary of the model you are trying to load is corrupted."
#             ):
#                 _ = ModelWithHead.from_pretrained(tmp_dir)

#     @require_torch_gpu
#     def test_pretrained_low_mem_new_config(self):
#         # Checking for 1 model(the same one which was described in the issue) .
#         model_ids = ["gpt2"]

#         for model_id in model_ids:
#             model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
#             model_config.n_layer = 48
#             model_config.n_head = 25
#             model_config.n_embd = 1600
#             model = AutoModelForCausalLM.from_pretrained(
#                 pretrained_model_name_or_path=model_id,
#                 config=model_config,
#                 ignore_mismatched_sizes=True,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#             model_ref = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)

#             self.assertEqual(model.__class__.__name__, model_ref.__class__.__name__)


# @require_torch
# @is_staging_test
# class ModelPushToHubTester(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls._token = TOKEN
#         set_access_token(TOKEN)
#         HfFolder.save_token(TOKEN)

#     @classmethod
#     def tearDownClass(cls):
#         try:
#             delete_repo(token=cls._token, repo_id="test-model")
#         except HTTPError:
#             pass

#         try:
#             delete_repo(token=cls._token, repo_id="valid_org/test-model-org")
#         except HTTPError:
#             pass

#         try:
#             delete_repo(token=cls._token, repo_id="test-dynamic-model")
#         except HTTPError:
#             pass

#     def test_push_to_hub(self):
#         config = BertConfig(
#             vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
#         )
#         model = BertModel(config)
#         model.push_to_hub("test-model", use_auth_token=self._token)

#         new_model = BertModel.from_pretrained(f"{USER}/test-model")
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#         # Reset repo
#         delete_repo(token=self._token, repo_id="test-model")

#         # Push to hub via save_pretrained
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(tmp_dir, repo_id="test-model", push_to_hub=True, use_auth_token=self._token)

#         new_model = BertModel.from_pretrained(f"{USER}/test-model")
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#     def test_push_to_hub_in_organization(self):
#         config = BertConfig(
#             vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
#         )
#         model = BertModel(config)
#         model.push_to_hub("valid_org/test-model-org", use_auth_token=self._token)

#         new_model = BertModel.from_pretrained("valid_org/test-model-org")
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#         # Reset repo
#         delete_repo(token=self._token, repo_id="valid_org/test-model-org")

#         # Push to hub via save_pretrained
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             model.save_pretrained(
#                 tmp_dir, push_to_hub=True, use_auth_token=self._token, repo_id="valid_org/test-model-org"
#             )

#         new_model = BertModel.from_pretrained("valid_org/test-model-org")
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#     def test_push_to_hub_dynamic_model(self):
#         CustomConfig.register_for_auto_class()
#         CustomModel.register_for_auto_class()

#         config = CustomConfig(hidden_size=32)
#         model = CustomModel(config)

#         model.push_to_hub("test-dynamic-model", use_auth_token=self._token)
#         # checks
#         self.assertDictEqual(
#             config.auto_map,
#             {"AutoConfig": "custom_configuration.CustomConfig", "AutoModel": "custom_modeling.CustomModel"},
#         )

#         new_model = AutoModel.from_pretrained(f"{USER}/test-dynamic-model", trust_remote_code=True)
#         # Can't make an isinstance check because the new_model is from the CustomModel class of a dynamic module
#         self.assertEqual(new_model.__class__.__name__, "CustomModel")
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#         config = AutoConfig.from_pretrained(f"{USER}/test-dynamic-model", trust_remote_code=True)
#         new_model = AutoModel.from_config(config, trust_remote_code=True)
#         self.assertEqual(new_model.__class__.__name__, "CustomModel")