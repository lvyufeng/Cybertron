import mindspore
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal
from ..common.ops import *
from ..common.layers import Dense
from ..common.activations import activation_map, GELU

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

    def rel_shift(x, klen=-1):
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
        ac = ibnd_jbnd_to_ijbn(q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = ibnd_jbnd_to_ijbn(q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = ibnd_snd_to_ibns(q_head + self.r_s_bias, self.seg_embed)
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
        attn_prob = ops.Softmax(-1)(attn_score)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # attention output
        attn_vec = ijbn_jbnd_to_ibnd(attn_prob, v_head_h)

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
        if g is not None:
            if mems is not None and mems.ndim > 1:
                cat = mnp.concatenate([mems, h])
            else:
                cat = h
            k_head_h = ops.dot(cat, self.k)
            v_head_h = ops.dot(cat, self.v)
            k_head_r = ops.dot(r, self.r)
            q_head_h = ops.dot(h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            
            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            
            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = ops.dot(g, self.q)

            # core attention ops
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
            if mems is not None and mems.dim() > 1:
                cat = mnp.concatenate([mems, h])
            else:
                cat = h

            # content heads
            q_head_h = ops.dot(h, self.q)
            k_head_h = ops.dot(cat, self.k)
            v_head_h = ops.dot(cat, self.v)

            # positional heads
            k_head_r = ops.dot(r, self.r)

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
        self.activation_function = activation_map.get(ff_activation, GELU())

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

