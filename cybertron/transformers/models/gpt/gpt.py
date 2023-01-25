import os
import logging
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, Normal
from cybertron.abc import PretrainedCell
from ...common.layers import activation_map, Dense, SequenceSummary
from ...configs.gpt import GPTConfig

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "openai-gpt": "https://huggingface.co/lvyufeng/gpt/resolve/main/openai-gpt.ckpt"
}

PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST = ["openai-gpt"]

def torch_to_mindspore(state_dict):
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'ln' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embed' in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    return ms_ckpt

class Conv1D(nn.Cell):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = Parameter(initializer(Normal(0.02), (nx, nf)), 'weight')
        self.bias = Parameter(initializer('zeros', nf), 'bias')
    
    def construct(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = ops.matmul(x.view(-1, x.shape[-1]), self.weight) + self.bias
        x = x.view(size_out)
        return x

class MLP(nn.Cell):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state) 
        afn = config.afn if config.afn != 'gelu' else 'gelu_approximate'
        self.act = activation_map.get(afn, nn.GELU())

    def construct(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Attention(nn.Cell):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx
        assert n_state % config.n_head == 0
        self.bias = Parameter(Tensor(np.tril(np.ones((n_ctx, n_ctx))), mindspore.float32).view(1, 1, n_ctx, n_ctx), 'bias')
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.output_attentions = config.output_attentions

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(1-config.attn_pdrop)
        self.resid_dropout = nn.Dropout(1-config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = ops.matmul(q, k)
        if self.scale:
            w = w / ops.sqrt(ops.scalar_to_tensor(v.shape[-1]))
        b = self.bias[:, :, : w.shape[-2], : w.shape[-1]]
        w = w * b + -1e9 * (1 - b)

        if attention_mask is not None:
            w = w + attention_mask

        w = ops.softmax(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask
        
        outputs = (ops.matmul(w, v),)
        if self.output_attentions:
            outputs += (w,)
        return outputs

    def split_heads(self, x, k=False):
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(new_x_shape)
        if k:
            return x.transpose(0, 2, 3, 1)
        else:
            return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.transpose(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(new_x_shape)

    def construct(self, x, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = ops.split(x, self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = (a,) + attn_outputs[1:]
        return outputs

class Block(nn.Cell):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm((nx,), epsilon=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm((nx,), epsilon=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def construct(self, x, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        a = output_attn[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = (h,) + output_attn[1:]
        return outputs

class GPTPretrainedCell(PretrainedCell):
    pretrained_model_archive = PRETRAINED_MODEL_ARCHIVE_MAP
    pytorch_pretrained_model_archive_list = PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST
    config_class = GPTConfig
    name = 'gpt'

class GPTModel(GPTPretrainedCell):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.n_layer = config.n_layer

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(1-config.embd_pdrop)
        self.h = nn.CellList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if position_ids is None:
            position_ids = ops.arange(input_ids.shape[-1], dtype=mindspore.int32)
            position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        
        if attention_mask is not None:
            attention_mask = attention_mask.expand_dims(1).expand_dims(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
                head_mask = ops.broadcast_to(head_mask, (self.n_layer, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = (None,) * self.n_layer
        
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        position_ids = position_ids.view(-1, position_ids.shape[-1])

        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(output_shape),)

            outputs = block(hidden_states, attention_mask, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(output_shape),)

        outputs = (hidden_states.view(output_shape),)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden states), (all attentions)

class GPTLMHeadModel(GPTPretrainedCell):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = Dense(config.n_embd, config.vocab_size, has_bias=False)

        self.lm_head.weight = self.transformer.tokens_embed.embedding_table

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]),
                                     shift_labels.view(-1), ignore_index=-1)
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, (all hidden states), (all attentions)

class GPTDoubleHeadsModel(GPTPretrainedCell):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = Dense(config.n_embd, config.vocab_size, has_bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.lm_head.weight = self.transformer.tokens_embed.embedding_table

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  mc_token_ids=None, lm_labels=None, mc_labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss = ops.cross_entropy(mc_logits.view(-1, mc_logits.shape[-1]),
                            mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = lm_labels[..., 1:]
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]),
                                     shift_labels.view(-1), ignore_index=-1)
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, (all hidden_states), (attentions)
