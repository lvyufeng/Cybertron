import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from typing import Optional
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform, _calculate_fan_in_and_fan_out
from mindspore import Tensor
from .ops import blh_bl_to_bh

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None):
        if weight_init is None:
            weight_init = initializer(HeUniform(math.sqrt(5)), (out_channels, in_channels))
        if bias_init is None:
            fan_in, _ = _calculate_fan_in_and_fan_out((out_channels, in_channels))
            bound = 1 / math.sqrt(fan_in)
            bias_init = initializer(Uniform(bound), (out_channels))
        super().__init__(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init, has_bias=has_bias, activation=activation)

class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32, padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings:Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding

class SequenceSummary(nn.Cell):
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = ops.Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = Dense(config.hidden_size, num_classes)

        self.activation = ops.Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()

        self.first_dropout = ops.Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = ops.Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(1-config.summary_last_dropout)

    def construct(self, hidden_states, cls_index=None):
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(axis=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = mnp.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=mindspore.int32)
            else:
                cls_index = cls_index.expand_dims(-1).expand_dims(-1)
                cls_index = mnp.broadcast_to(cls_index, (-1,) * (cls_index.ndim-1) + (hidden_states.shape[-1],))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = ops.gather_d(hidden_states, -2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

class PoolerStartLogits(nn.Cell):
    """ Compute SQuAD start_logits from sequence hidden states. """
    def __init__(self, config):
        super(PoolerStartLogits, self).__init__()
        self.dense = Dense(config.hidden_size, 1)

    def construct(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if hidden_states.dtype == mindspore.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Cell):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """
    def __init__(self, config):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dense_1 = Dense(config.hidden_size, 1)

    def construct(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.
            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = mnp.broadcast_to(start_positions[:, None, None], (-1, -1, hsz)) # shape (bsz, 1, hsz)
            start_states = ops.gather_d(hidden_states, -2, start_positions) # shape (bsz, 1, hsz)
            start_states = mnp.broadcast_to(start_states, (-1, slen, -1)) # shape (bsz, slen, hsz)

        x = self.dense_0(mnp.concatenate((hidden_states, start_states), axis=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Cell):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, config):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = Dense(config.hidden_size, 1, bias=False)

    def construct(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.
            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.
            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = mnp.broadcast_to(start_positions[:, None, None], (-1, -1, hsz)) # shape (bsz, 1, hsz)
            start_states = ops.gather_d(hidden_states, -2, start_positions).squeeze(-2) # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = mnp.broadcast_to(cls_index[:, None, None], (-1, -1, hsz)) # shape (bsz, 1, hsz)
            cls_token_state = ops.gather_d(hidden_states, -2, cls_index).squeeze(-2) # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :] # shape (bsz, hsz)

        x = self.dense_0(mnp.concatenate((start_states, cls_token_state), axis=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


class SQuADHead(nn.Cell):
    r""" A SQuAD head inspired by XLNet.
    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """
    def __init__(self, config):
        super(SQuADHead, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def construct(self, hidden_states, start_positions=None, end_positions=None,
                  cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()

        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.ndim > 1:
                    x.squeeze(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.shape
            start_log_probs = ops.Softmax()(start_logits) # shape (bsz, slen)

            start_top_log_probs, start_top_index = ops.TopK()(start_log_probs, self.start_n_top) # shape (bsz, start_n_top)
            start_top_index_exp = mnp.broadcast_to(start_top_index.expand_dims(-1), (-1, -1, hsz)) # shape (bsz, start_n_top, hsz)
            start_states = ops.gather_d(hidden_states, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
            start_states = mnp.broadcast_to(start_states.expand_dims(1), (-1, slen, -1, -1)) # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.expand_dims(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.expand_dims(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = ops.Softmax(1)(end_logits) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = ops.TopK()(end_log_probs, self.end_n_top) # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = blh_bl_to_bh(hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(False),
    'gelu_approximate': nn.GELU(),
    'swish':nn.SiLU()
}