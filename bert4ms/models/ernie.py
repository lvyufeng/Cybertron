# class ErnieModel(nn.Cell):
#     """
#     Bidirectional Encoder Representations from Transformers.
#     Args:
#         config (Class): Configuration for ErnieModel.
#         is_training (bool): True for training mode. False for eval mode.
#         use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
#     """
#     def __init__(self,
#                  config,
#                  is_training,
#                  use_one_hot_embeddings=False):
#         super(ErnieModel, self).__init__()
#         config = copy.deepcopy(config)
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0

#         self.seq_length = config.seq_length
#         self.hidden_size = config.hidden_size
#         self.num_hidden_layers = config.num_hidden_layers
#         self.embedding_size = config.hidden_size
#         self.token_type_ids = None

#         self.last_idx = self.num_hidden_layers - 1
#         output_embedding_shape = [-1, self.seq_length, self.embedding_size]

#         self.ernie_embedding_lookup = nn.Embedding(
#             vocab_size=config.vocab_size,
#             embedding_size=self.embedding_size,
#             use_one_hot=use_one_hot_embeddings)

#         self.ernie_embedding_postprocessor = EmbeddingPostprocessor(
#             embedding_size=self.embedding_size,
#             embedding_shape=output_embedding_shape,
#             use_relative_positions=config.use_relative_positions,
#             use_token_type=True,
#             token_type_vocab_size=config.type_vocab_size,
#             use_one_hot_embeddings=use_one_hot_embeddings,
#             initializer_range=0.02,
#             max_position_embeddings=config.max_position_embeddings,
#             dropout_prob=config.hidden_dropout_prob)

#         self.ernie_encoder = ErnieTransformer(
#             hidden_size=self.hidden_size,
#             seq_length=self.seq_length,
#             num_attention_heads=config.num_attention_heads,
#             num_hidden_layers=self.num_hidden_layers,
#             intermediate_size=config.intermediate_size,
#             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
#             use_one_hot_embeddings=use_one_hot_embeddings,
#             initializer_range=config.initializer_range,
#             hidden_dropout_prob=config.hidden_dropout_prob,
#             use_relative_positions=config.use_relative_positions,
#             hidden_act=config.hidden_act,
#             compute_type=config.compute_type,
#             return_all_encoders=True)

#         self.cast = P.Cast()
#         self.dtype = config.dtype
#         self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
#         self.slice = P.StridedSlice()

#         self.squeeze_1 = P.Squeeze(axis=1)
#         self.dense = nn.Dense(self.hidden_size, self.hidden_size,
#                               activation="tanh",
#                               weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
#         self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

#     def construct(self, input_ids, token_type_ids, input_mask):
#         """Bidirectional Encoder Representations from Transformers."""
#         # embedding
#         word_embeddings = self.ernie_embedding_lookup(input_ids)
#         embedding_output = self.ernie_embedding_postprocessor(token_type_ids,
#                                                               word_embeddings)

#         # attention mask [batch_size, seq_length, seq_length]
#         attention_mask = self._create_attention_mask_from_input_mask(input_mask)

#         # ernie encoder
#         encoder_output = self.ernie_encoder(self.cast_compute_type(embedding_output),
#                                             attention_mask)

#         sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

#         # pooler
#         batch_size = P.Shape()(input_ids)[0]
#         sequence_slice = self.slice(sequence_output,
#                                     (0, 0, 0),
#                                     (batch_size, 1, self.hidden_size),
#                                     (1, 1, 1))
#         first_token = self.squeeze_1(sequence_slice)
#         pooled_output = self.dense(first_token)
#         pooled_output = self.cast(pooled_output, self.dtype)

#         return sequence_output, pooled_output