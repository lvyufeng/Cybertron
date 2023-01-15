import os
import logging
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, Normal
from cybertron.abc import PretrainedCell
from ..common.modules.ops import ibnd_snd_to_ibns, ijbs_ibns_to_ijbn, mbnd_mlb_to_lbnd, lbnd_mlb_to_mbnd, blh_bl_to_bh
from ..common.modules import Dense, SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, \
    activation_map
from ..configs.xlnet import XLNetConfig

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlnet-base-cased": "https://huggingface.co/lvyufeng/xlnet/resolve/main/xlnet-base-cased.ckpt", 
    "xlnet-large-cased": "https://huggingface.co/lvyufeng/xlnet/resolve/main/xlnet-large-cased.ckpt"
}

PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST = ["xlnet-base-cased", "xlnet-large-cased"]

def torch_to_mindspore(state_dict):
    ms_ckpt = []

    for k, v in state_dict.items():
        if 'layer_norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embedding' in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    return ms_ckpt

class XLNetRelativeAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head))

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = Parameter(initializer(Normal(config.initializer_range), (config.d_model, self.n_head, self.d_head)), 'q')
        self.k = Parameter(initializer(Normal(config.initializer_range), (config.d_model, self.n_head, self.d_head)), 'k')
        self.v = Parameter(initializer(Normal(config.initializer_range), (config.d_model, self.n_head, self.d_head)), 'v')
        self.o = Parameter(initializer(Normal(config.initializer_range), (config.d_model, self.n_head, self.d_head)), 'o')
        self.r = Parameter(initializer(Normal(config.initializer_range), (config.d_model, self.n_head, self.d_head)), 'r')

        self.r_r_bias = Parameter(initializer(Normal(config.initializer_range), (self.n_head, self.d_head)), 'r_r_bias')
        self.r_s_bias = Parameter(initializer(Normal(config.initializer_range), (self.n_head, self.d_head)), 'r_s_bias')
        self.r_w_bias = Parameter(initializer(Normal(config.initializer_range), (self.n_head, self.d_head)), 'r_w_bias')

        self.seg_embed = Parameter(initializer(Normal(config.initializer_range), (2, self.n_head, self.d_head)), 'seg_embed')
        self.layer_norm = nn.LayerNorm((config.d_model,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1-config.dropout)

    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = x[:, 0:klen, :, :]

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""
        # content based attention score
        # ibnd_jbnd_to_ijbn
        ac = ops.bmm((q_head + self.r_w_bias).transpose(1, 2, 0, 3), k_head_h.transpose(1, 2, 3, 0)).transpose(2, 3, 0, 1)
        # position based attention score
        # ibnd_jbnd_to_ijbn
        bd = ops.bmm((q_head + self.r_r_bias).transpose(1, 2, 0, 3), k_head_r.transpose(1, 2, 3, 0)).transpose(2, 3, 0, 1)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = ibnd_snd_to_ibns(q_head + self.r_s_bias, self.seg_embed)
            # ijbs_ibns_to_ijbn
            ef = ijbs_ibns_to_ijbn(seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == mindspore.float16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = ops.softmax(attn_score, -1)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # attention output
        # ijbn_jbnd_to_ibnd
        attn_vec = ops.bmm(attn_prob.transpose(2, 3, 0, 1), v_head_h.transpose(1, 2, 0, 3)).transpose(2, 0, 1, 3)

        if self.output_attentions:
            return attn_vec, attn_prob

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = ops.tensor_dot(attn_vec, self.o, ((2, 3), (1, 2)))

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def construct(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, 
                  mems=None, target_mapping=None, head_mask=None):
        attn_prob = None
        if g is not None:
            if mems is not None and mems.ndim > 1:
                cat = ops.concat([mems, h])
            else:
                cat = h
            k_head_h = ops.dot(cat, self.k)
            v_head_h = ops.dot(cat, self.v)
            k_head_r = ops.dot(r, self.r)
            q_head_h = ops.dot(h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            attn_prob_h = None

            if self.output_attentions:
                (attn_vec_h, attn_prob_h) = attn_vec_h
            
            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = ops.dot(g, self.q)

            # core attention ops
            attn_prob_g = None
            if target_mapping is not None:
                q_head_g = mbnd_mlb_to_lbnd(q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = lbnd_mlb_to_mbnd(attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            ###### Multi-head attention with relative positional encoding
            if mems is not None and mems.ndim > 1:
                cat = ops.concat([mems, h])
            else:
                cat = h

            # content heads
            q_head_h = ops.tensor_dot(h, self.q, ((2,), (0,)))
            k_head_h = ops.tensor_dot(cat, self.k, ((2,), (0,)))
            v_head_h = ops.tensor_dot(cat, self.v, ((2,), (0,)))

            # positional heads
            k_head_r = ops.tensor_dot(r, self.r, ((2,), (0,)))

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForward(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm((config.d_model,), epsilon=config.layer_norm_eps)
        self.layer_1 = Dense(config.d_model, config.d_inner)
        self.layer_2 = Dense(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(1-config.dropout)
        ff_activation = 'gelu_approximate' if config.ff_activation == 'gelu' else config.ff_activation
        self.activation_function = activation_map.get(ff_activation, nn.GELU())

    def construct(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class XLNetLayer(nn.Cell):
    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(1-config.dropout)

    def construct(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat,
                  mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g,
                                r, seg_mat, mems=mems, target_mapping=target_mapping,
                                head_mask=head_mask)
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs

class XLNetPretrainedCell(PretrainedCell):
    pretrained_model_archive = PRETRAINED_MODEL_ARCHIVE_MAP
    pytorch_pretrained_model_archive_list = PYTORCH_PRETRAINED_MODEL_ARCHIVE_LIST
    config_class = XLNetConfig
    name = 'xlnet'

class XLNetModel(XLNetPretrainedCell):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.n_token, config.d_model)
        self.mask_emb = Parameter(initializer(Normal(config.initializer_range), (1, 1, config.d_model)), "mask_emb")
        self.layer = nn.CellList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(1-config.dropout)

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill
        ::
                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = ops.ones((qlen, qlen))
        mask_up = ops.triu(attn_mask, k=1)
        attn_mask_pad = ops.zeros((qlen, mlen))
        ret = ops.concatenate((attn_mask_pad, mask_up), axis=1)
        if self.same_length:
            mask_lo = ops.tril(attn_mask, k=-1)
            ret = ops.concat((ret[:, :qlen] + mask_lo, ret[:, qlen:]), axis=1)
        return ret

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        sinusoid_inp = ops.matmul(pos_seq.expand_dims(-1), inv_freq.expand_dims(0))
        pos_emb = ops.concat((ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)), axis=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = ops.BroadcastTo((-1, bsz, -1))(pos_emb)
            # pos_emb = ops.broadcast_to(pos_emb, (-1, bsz, -1))
            # pos_emb = pos_emb.expand(Tensor((-1, bsz, -1)))
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = ops.arange(0, self.d_model, 2, dtype=mindspore.float32)
        inv_freq = 1 / ops.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            beg, end = klen, -1
        # else:
        #     raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = ops.arange(beg, end, -1.0, dtype=mindspore.float32)
            bwd_pos_seq = ops.arange(-beg, -end, 1.0, dtype=mindspore.float32)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clip(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clip(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = ops.concat((fwd_pos_emb, bwd_pos_emb), axis=1)
        else:
            fwd_pos_seq = ops.arange(beg, end, -1)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clip(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.mem_len is None or self.mem_len == 0:
            return None
        else:
            if self.reuse_len is not None and self.reuse_len > 0:
                curr_out = curr_out[:self.reuse_len]

            if prev_mem is None:
                new_mem = curr_out[-self.mem_len:]
            else:
                new_mem = ops.concat((prev_mem, curr_out), axis=0)[-self.mem_len:]

        return new_mem

    def construct(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                  token_type_ids=None, input_mask=None, head_mask=None):
        input_ids = input_ids.swapaxes(0, 1)
        token_type_ids = token_type_ids.swapaxes(0, 1) if token_type_ids is not None else None
        input_mask = input_mask.swapaxes(0, 1) if input_mask is not None else None
        attention_mask = attention_mask.swapaxes(0, 1) if attention_mask is not None else None
        perm_mask = perm_mask.transpose(1, 2, 0) if perm_mask is not None else None
        target_mapping = target_mapping.transpose(1, 2, 0) if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            attn_mask = None
        # data mask: input mask & perm mask
        # assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        # "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = ops.zeros((data_mask.shape[0], mlen, bsz))
                data_mask = ops.concat((mems_mask, data_mask), axis=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0)

        if attn_mask is not None:
            non_tgt_mask = -ops.eye(qlen)
            if mlen > 0:
                non_tgt_mask = ops.cat((ops.zeros((qlen, mlen)), non_tgt_mask), axis=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = ops.broadcast_to(self.mask_emb, (target_mapping.shape[0], bsz, -1))
        # else:  # We removed the inp_q input which was same as target mapping
        #     inp_q_ext = inp_q[:, :, None]
        #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = ops.zeros((mlen, bsz), dtype=mindspore.int32)
                cat_ids = ops.concat((mem_pad, token_type_ids), axis=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).astype(mindspore.int32)
            seg_mat = ops.one_hot(seg_mat, seg_mat.ndim + 2, ops.scalar_to_tensor(1, mindspore.int32), ops.scalar_to_tensor(0, mindspore.int32))
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(0).expand_dims(0)
                head_mask = ops.broadcast_to(head_mask, (self.n_layer, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(1).expand_dims(1)
        else:
            head_mask = (None,) * self.n_layer

        new_mems = ()
        if mems is None:
            mems = (None,) * len(self.layer)

        attentions = ()
        hidden_states = ()
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                if output_g is not None:
                    hidden_states += (output_h, output_g)
                else:
                    hidden_states += (output_h,)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions += (outputs[2],)

        # Add last hidden state
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states += (output_h, output_g)
            else:
                hidden_states += (output_h,)
        if output_g is not None:
            output = self.dropout(output_g)
        else:
            output = self.dropout(output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.transpose(1, 0, 2), new_mems)
        new_hidden_states = ()
        if self.output_hidden_states:
            if output_g is not None:
                for hs in hidden_states:
                    for h in hs:
                        new_hidden_states += (h.transpose(1, 0, 2),)
            else:
                for hs in hidden_states:
                    new_hidden_states += (hs.transpose(1, 0, 2),)
            outputs = outputs + (new_hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.transpose(2, 3, 0, 1) for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, new_mems, (hidden_states), (attentions)

class XLNetLMHeadModel(XLNetPretrainedCell):
    def __init__(self, config):
        super(XLNetLMHeadModel, self).__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        self.transformer = XLNetModel(config)
        self.lm_loss = Dense(config.d_model, config.n_token, has_bias=True)

        self.lm_loss.weight = self.transformer.word_embedding.embedding_table

    def construct(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                  token_type_ids=None, input_mask=None, head_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask)

        logits = self.lm_loss(transformer_outputs[0])

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            # Flatten the tokens
            loss = ops.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, mems, (hidden states), (attentions)

class XLNetForSequenceClassification(XLNetPretrainedCell):
    def __init__(self, config):
        super(XLNetForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = Dense(config.d_model, config.num_labels)

    def construct(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                  token_type_ids=None, input_mask=None, head_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask)
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss = ops.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, mems, (hidden states), (attentions)

class XLNetForMultipleChoice(XLNetPretrainedCell):
    def __init__(self, config):
        super(XLNetForMultipleChoice, self).__init__(config)

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = Dense(config.d_model, 1)

    def construct(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                  mems=None, perm_mask=None, target_mapping=None, labels=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.shape[-1]) if input_mask is not None else None

        transformer_outputs = self.transformer(flat_input_ids, token_type_ids=flat_token_type_ids,
                                               input_mask=flat_input_mask, attention_mask=flat_attention_mask,
                                               mems=mems, perm_mask=perm_mask, target_mapping=target_mapping,
                                               head_mask=head_mask)


        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, mems, (hidden states), (attentions)

class XLNetForQuestionAnsweringSimple(XLNetPretrainedCell):
    def __init__(self, config):
        super(XLNetForQuestionAnsweringSimple, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.qa_outputs = Dense(config.hidden_size, config.num_labels)

    def construct(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                  token_type_ids=None, input_mask=None, head_mask=None,
                  start_positions=None, end_positions=None):

        outputs = self.transformer(input_ids,
                                   attention_mask=attention_mask,
                                   mems=mems,
                                   perm_mask=perm_mask,
                                   target_mapping=target_mapping,
                                   token_type_ids=token_type_ids,
                                   input_mask=input_mask, 
                                   head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = ops.split(logits, 1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions.clip(0, ignored_index)
            end_positions.clip(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class XLNetForQuestionAnswering(XLNetPretrainedCell):
    def __init__(self, config):
        super(XLNetForQuestionAnswering, self).__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None,
                start_positions=None, end_positions=None, is_impossible=None, cls_index=None, p_mask=None,):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            start_loss = ops.cross_entropy(start_logits, start_positions)
            end_loss = ops.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                cls_loss = ops.binary_cross_entropy_with_logits(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.shape
            start_log_probs = ops.softmax(start_logits) # shape (bsz, slen)

            start_top_log_probs, start_top_index = ops.top_k(start_log_probs, self.start_n_top) # shape (bsz, start_n_top)
            start_top_index_exp = ops.broadcast_to(start_top_index.expand_dims(-1), (-1, -1, hsz)) # shape (bsz, start_n_top, hsz)
            start_states = ops.gather_d(hidden_states, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
            start_states = ops.broadcast_to(start_states.expand_dims(1), (-1, slen, -1, -1)) # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.expand_dims(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.expand_dims(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = ops.softmax(end_logits, 1) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = ops.top_k(end_log_probs, self.end_n_top) # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = blh_bl_to_bh(hidden_states, start_log_probs)  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs
