import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.nc_models import BaseClassifyModel
# from model.distill_bert import DistilBertModel
from models.layers import AttLayer, MultiHeadedAttention, AddNorm, GRUWithAttention, RNNBase


#
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.variant_name == "gru" or self.variant_name == "combined_gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         if self.variant_name == "weight_mha":
#             head_dim = self.embed_dim // 12
#             self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
#         if self.variant_name == "combined_mha":
#             self.query = nn.Linear(self.embed_dim, topic_dim)
#             self.key = nn.Linear(self.embed_dim, topic_dim)
#         if self.variant_name == "reuse":
#             self.projection = self.topic_layer
#         else:
#             self.projection = AttLayer(self.embed_dim, 128)
#
#     def run_gru(self, embedding, length):
#         try:
#             embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         except RuntimeError:
#             raise RuntimeError()
#         y, _ = self.gru(embedding)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         return y
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
#         topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
#         # expand mask to the same size as topic weights
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
#         topic_vec, topic_weight = self.extract_topic(input_feat)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H) (32,768), (32,8,1)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output

# 平均 0.780595 python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove --head_num=180 -ce=1 -lr=0.001 -vn=combined_mha -ep=5 -dr=0.2
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
#         if self.variant_name == "topic_embed":
#             topic_weight = self.topic_layer(input_feat["data"]).transpose(1, 2)  # (N, H, S)
#         else:
#             topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
#         # expand mask to the same size as topic weights
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         if self.variant_name == "combined_mha":
#             # context_vec = torch.matmul(topic_weight, embedding)  # (N, H, E)
#             query, key = [linear(embedding).view(embedding.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
#                           for linear in (self.query, self.key)]
#             # topic_vec, _ = self.mha(context_vec, context_vec, context_vec)  # (N, H, H*D)
#             scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_num ** 0.5  # (N, H, S, S)
#             context_weight = torch.mean(scores, dim=-1)  # (N, H, S)
#             topic_weight = context_weight * topic_weight  # (N, H, S)
#         elif self.variant_name == "combined_gru":
#             length = torch.sum(input_feat["mask"], dim=-1)
#             embedding = self.run_gru(embedding, length)
#         elif self.variant_name == "weight_mha":
#             embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
#         return topic_vec, topic_weight

    # def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    #     input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
    #     topic_vec, topic_weight = self.extract_topic(input_feat)
    #     if self.variant_name == "reuse":
    #         doc_topic = torch.mean(self.topic_layer(topic_vec), -1).unsqueeze(-1)  # (N, H)
    #         doc_embedding = torch.sum(topic_vec * doc_topic, dim=1)  # (N, E)
    #     else:
    #         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H) (32,768), (32,8,1)
    #     output = self.classify_layer(doc_embedding, topic_weight, return_attention)
    #     if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
    #         entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
    #         output = output + (entropy_sum,)
    #     return output



# BiLSTM,还行
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         self.batch_size = kwargs.pop("batch_size",32)
#         self.max_length = kwargs.pop("max_length", 100)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(2 * self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(2 * self.embed_dim, self.head_num)
#         if self.variant_name == "gru" or self.variant_name == "combined_gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         if self.variant_name == "weight_mha":
#             head_dim = self.embed_dim // 12
#             self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
#         if self.variant_name == "combined_mha":
#             if self.with_gru == "gru":
#                 self.gru =  nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#             elif self.with_gru == "biLSTM":
#                 self.gru = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True, bidirectional=True)
#             # self.mha1 = nn.MultiheadAttention(self.embed_dim, self.head_num,batch_first=True)
#             # self.mha2 = nn.MultiheadAttention(self.embed_dim, self.head_num, batch_first=True)
#             # self.mha3 = nn.MultiheadAttention(self.embed_dim, self.head_num, batch_first=True)
#             # self.add_norm1 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#             # self.add_norm2 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#             # self.add_norm3 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#             # self.feature_network1 = nn.Linear(self.embed_dim, self.embed_dim)
#             # self.feature_network2 = nn.Linear(self.embed_dim, self.embed_dim)
#             # self.feature_network3 = nn.Linear(self.embed_dim, self.embed_dim)
#             # self.query = nn.Linear(self.embed_dim, topic_dim)
#             # # self.key = nn.Linear(self.embed_dim, topic_dim)
#             # self.W_o = nn.Linear(topic_dim, topic_dim)
#             self.mha1 = nn.MultiheadAttention(2 * self.embed_dim, self.head_num,batch_first=True)
#             self.add_norm1 = AddNorm([self.max_length, 2 * self.embed_dim], self.dropout_rate)
#         if self.variant_name == "reuse":
#             self.projection = self.topic_layer
#         else:
#             self.projection = AttLayer(self.embed_dim, 128)
#
#     def run_gru(self, embedding, length):
#         try:
#             embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         except RuntimeError:
#             raise RuntimeError()
#         y, _ = self.gru(embedding)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         return y
#     # python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove -ce=1 -lr=0.001 -vn=combined_mha -ep=8 -dr=0.5 -hn=10 -hd=30 -wg=gru
#     # 0.782022
#     # def extract_topic(self, input_feat):
#     #     embedding = self.embedding_layer(input_feat)  # (N, S) -> (N,S,E) 即 (32,100)->(32,100,768)
#     #     if self.with_gru:
#     #         length = torch.sum(input_feat["mask"], dim=-1)
#     #         x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#     #         y, _ = self.gru(x)  # extract interest from history behavior
#     #         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#     #         temp_vec = nn.Dropout(self.dropout_rate)(y)
#     #     temp_vec1, _ = self.mha1(temp_vec, temp_vec, temp_vec)  # (N, S, H*D=E)
#     #     topic_vec = self.add_norm1(temp_vec, temp_vec1)
#     #     if self.variant_name == "topic_embed":
#     #         topic_weight = self.topic_layer(input_feat["data"]).transpose(1, 2)  # (N, H, S)
#     #     else:
#     #         # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#     #         topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#     #         # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#     #     # expand mask to the same size as topic weights
#     #     mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0 # mask: torch.Size([32, 180, 100])
#     #     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#     #     # topic_vec=None
#     #     if self.variant_name == "combined_mha":
#     #         pass
#     #     elif self.variant_name == "combined_gru":
#     #         length = torch.sum(input_feat["mask"], dim=-1)
#     #         embedding = self.run_gru(embedding, length)
#     #     elif self.variant_name == "weight_mha":
#     #         embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
#     #     topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # （N, H, S） x (N, S, E)
#     #     return topic_vec, topic_weight
#
# 平均 0.783157  python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove --head_num=10 -ce=1 -lr=0.001 -vn=combined_mha -ep=5 -dr=0.5 -wg=gru
# def extract_topic(self, input_feat):
#     embedding = self.embedding_layer(input_feat)  # (N, S) -> (N,S,E) 即 (32,100)->(32,100,768)
#     if self.with_gru:
#         length = torch.sum(input_feat["mask"], dim=-1)
#         x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         y, _ = self.gru(x)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         temp_vec = nn.Dropout(self.dropout_rate)(y)
#     temp_vec1, _ = self.mha1(embedding, temp_vec, temp_vec)  # (N, S, H*D=E)
#     topic_vec = self.add_norm1(embedding, temp_vec1)
#     if self.variant_name == "topic_embed":
#         topic_weight = self.topic_layer(input_feat["data"]).transpose(1, 2)  # (N, H, S)
#     else:
#         # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#         topic_weight = self.topic_layer(embedding).transpose(1,
#                                                              2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#         # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#     # expand mask to the same size as topic weights
#     mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0,
#                                                                                      1) == 0  # mask: torch.Size([32, 180, 100])
#     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#     # topic_vec=None
#     if self.variant_name == "combined_mha":
#         pass
#     elif self.variant_name == "combined_gru":
#         length = torch.sum(input_feat["mask"], dim=-1)
#         embedding = self.run_gru(embedding, length)
#     elif self.variant_name == "weight_mha":
#         embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
#     topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # （N, H, S） x (N, S, E)
#     return topic_vec, topic_weight

# # BiLSTM  python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove -ce=1 -lr=0.001 -vn=combined_mha -ep=5 -dr=0.5 -hn=10  -wg=biLSTM
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N,S,E) 即 (32,100)->(32,100,768)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, _ = self.gru(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             temp_vec = nn.Dropout(self.dropout_rate)(y)
#         temp_vec1, _ = self.mha1(temp_vec, temp_vec, temp_vec)  # (N, S, H*D=E)
#         topic_vec = self.add_norm1(temp_vec, temp_vec1)
#         if self.variant_name == "topic_embed":
#             topic_weight = self.topic_layer(input_feat["data"]).transpose(1, 2)  # (N, H, S)
#         else:
#             # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#             topic_weight = self.topic_layer(topic_vec).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#             # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#         # expand mask to the same size as topic weights
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0 # mask: torch.Size([32, 180, 100])
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         # topic_vec=None
#         if self.variant_name == "combined_mha":
#             pass
#         elif self.variant_name == "combined_gru":
#             length = torch.sum(input_feat["mask"], dim=-1)
#             embedding = self.run_gru(embedding, length)
#         elif self.variant_name == "weight_mha":
#             embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
#         topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # （N, H, S） x (N, S, E)
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat)
#         if self.variant_name == "reuse":
#             doc_topic = torch.mean(self.topic_layer(topic_vec), -1).unsqueeze(-1)  # (N, H)
#             doc_embedding = torch.sum(topic_vec * doc_topic, dim=1)  # (N, E)
#         else:
#             doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output

#
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         self.batch_size = kwargs.pop("batch_size",32)
#         self.max_length = kwargs.pop("max_length", 100)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.variant_name == "gru" or self.variant_name == "combined_gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         if self.variant_name == "weight_mha":
#             head_dim = self.embed_dim // 12
#             self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
#         if self.variant_name == "combined_mha":
#             pass
#         if self.with_gru == "gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True)
#         elif self.with_gru == "biLSTM":
#             self.gru = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True, bidirectional=True)
#         self.mha1 = nn.MultiheadAttention(self.embed_dim, self.head_num, batch_first=True)
#         self.mha2 = nn.MultiheadAttention(self.embed_dim, self.head_num, batch_first=True)
#         self.mha3 = nn.MultiheadAttention(self.embed_dim, self.head_num, batch_first=True)
#         self.add_norm1 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#         self.add_norm2 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#         self.add_norm3 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#         self.feature_network1 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.feature_network2 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.feature_network3 = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name == "reuse":
#             self.projection = self.topic_layer
#         else:
#             self.projection = AttLayer(self.embed_dim, 128)
#
#     def run_gru(self, embedding, length):
#         try:
#             embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         except RuntimeError:
#             raise RuntimeError()
#         y, _ = self.gru(embedding)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         return y
#   #  python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove -ce=1 -lr=0.001 -vn=combined_mha -ep=8 -dr=0.5 -hn=10 -hd=30 -wg=gru
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N,S,E) 即 (32,100)->(32,100,768)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, _ = self.gru(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             temp_vec = nn.Dropout(self.dropout_rate)(y)
#         temp_vec1, _ = self.mha1(embedding, embedding, embedding)  # (N, S, H*D=E)
#         topic_vec = self.add_norm1(embedding, temp_vec1)
#         topic_vec += temp_vec
#         topic_weight = self.topic_layer(topic_vec).transpose(1,2)  # (N, S, E) -> （N, H, S） 即 (32,100, 768) -> (32, 180, 100)
#         # expand mask to the same size as topic weights
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0,1) == 0  # mask: torch.Size([32, 180, 100])
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # （N, H, S） x (N, S, E)
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat)
#         if self.variant_name == "reuse":
#             doc_topic = torch.mean(self.topic_layer(topic_vec), -1).unsqueeze(-1)  # (N, H)
#             doc_embedding = torch.sum(topic_vec * doc_topic, dim=1)  # (N, E)
#         else:
#             doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output




# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.variant_name == "gru" or self.variant_name == "combined_gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         if self.variant_name == "weight_mha":
#             head_dim = self.embed_dim // 12
#             self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
#         if self.variant_name == "combined_mha":
#             self.query = nn.Linear(self.embed_dim, topic_dim)
#             self.key = nn.Linear(self.embed_dim, topic_dim)
#         if self.variant_name == "reuse":
#             self.projection = self.topic_layer
#         else:
#             self.projection = AttLayer(self.embed_dim, 128)
#
#     def run_gru(self, embedding, length):
#         try:
#             embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         except RuntimeError:
#             raise RuntimeError()
#         y, _ = self.gru(embedding)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         return y
#
#     # Mean Pooling - Take attention mask into account for correct averaging
#     def mean_pooling(model_output, attention_mask):
#         token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
# # 平均 0.780595 python experiment/runner/nc_baseline.py --arch_type=BiAttentionClassifyModel -et=glove --head_num=180 -ce=1 -lr=0.001 -vn=combined_mha -ep=5 -dr=0.2
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)
#         topic_weight = self.topic_layer(embedding).transpose(1, 2)
#         # doc_embedding, word_embedding = input_feat['doc_embedding'], input_feat['word_embedding']
#         doc_embedding = input_feat['doc_embedding']
#         return doc_embedding, topic_weight
#
#     # def extract_keywords(self, input_feat):
#
#
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         doc_embedding, topic_weight = self.extract_topic(input_feat) # (32,768), (32,8,100)
#         if self.variant_name == "reuse":
#             doc_topic = torch.mean(self.topic_layer(doc_embedding), -1).unsqueeze(-1)  # (N, H)
#             doc_embedding = torch.sum(doc_embedding * doc_topic, dim=1)  # (N, E)
#         else:
#             doc_embedding, doc_topic = self.projection(doc_embedding)  # (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output
#



# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         topic_dim = self.head_num * self.head_dim
#         self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         self.mha1 = nn.MultiheadAttention(self.embed_dim, self.head_num,batch_first=True)
#         self.add_norm1 = AddNorm([self.max_length, self.embed_dim], self.dropout_rate)
#         self.feature_network1 = nn.Linear(self.embed_dim, self.embed_dim)
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.variant_name == "gru" or self.variant_name == "combined_gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
#         if self.variant_name == "weight_mha":
#             head_dim = self.embed_dim // 12
#             self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
#         if self.variant_name == "combined_mha":
#             self.query = nn.Linear(self.embed_dim, topic_dim)
#             self.key = nn.Linear(self.embed_dim, topic_dim)
#         if self.variant_name == "reuse":
#             self.projection = self.topic_layer
#         else:
#             self.projection = AttLayer(self.embed_dim, 128)
#
#     def run_gru(self, embedding, length):
#         try:
#             embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         except RuntimeError:
#             raise RuntimeError()
#         y, _ = self.gru(embedding)  # extract interest from history behavior
#         y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         return y

    # def extract_topic(self, input_feat):
    #     embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
    #     length = torch.sum(input_feat["mask"], dim=-1)
    #     y = self.run_gru(embedding, length)
    #     key_embedding = self.embedding(input_feat["key_word"], input_feat["mask_key"], inputs_embeds=input_feat["embedding"])[0]
    #     embedding = y + key_embedding
    #     topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
    #     # expand mask to the same size as topic weights
    #     mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
    #     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
    #     topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
    #     return topic_vec, topic_weight
    #
    # def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    #     input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
    #     topic_vec, topic_weight = self.extract_topic(input_feat)
    #     doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H) (32,768), (32,8,1)
    #     output = self.classify_layer(doc_embedding, topic_weight, return_attention)
    #     if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
    #         entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
    #         output = output + (entropy_sum,)
    #     return output



    # ml=256 时跑出 0.817
    # def extract_topic(self, input_feat):
    #     embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
    #     # length = torch.sum(input_feat["mask"], dim=-1)
    #     # y = self.run_gru(embedding, length)
    #     # topic_vec, _ = self.mha1(embedding,embedding,embedding)
    #     key_embedding = self.embedding(input_feat["key_word"], input_feat["mask_key"], inputs_embeds=input_feat["embedding"])[0]
    #     # topic_vec = self.add_norm1(key_embedding, topic_vec)
    #     topic_vec, _ = self.mha1(key_embedding, embedding, embedding)
    #     topic_vec = self.add_norm1(key_embedding, topic_vec)
    #     topic_weight1 = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
    #     topic_weight2 = self.topic_layer(key_embedding).transpose(1, 2)  # (N, H, S)
    #     topic_weight = topic_weight1 + topic_weight2
    #     # # expand mask to the same size as topic weights
    #     mask = input_feat["mask"].expand(self.head_num, topic_weight.size(0), -1).transpose(0, 1) == 0
    #     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
    #     topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # (N, H, E)
    #     return topic_vec, topic_weight
    #
    # def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    #     input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
    #     embedding, topic_weight = self.extract_topic(input_feat)
    #     doc_embedding, doc_topic = self.projection(embedding)  # (N, E), (N, H) (32,768), (32,8,1)
    #     output = self.classify_layer(doc_embedding, topic_weight, return_attention)
    #     if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
    #         entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
    #         output = output + (entropy_sum,)
    #     return output

    # def extract_topic(self, input_feat):
    #     embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
    #     key_embedding = self.embedding(input_feat["key_word"], input_feat["mask_key"], inputs_embeds=input_feat["embedding"])[0]
    #     topic_vec, _ = self.mha1(key_embedding, embedding, embedding)
    #     topic_vec = self.add_norm1(embedding, topic_vec)
    #     # topic_weight1 = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
    #     # topic_weight2 = self.topic_layer(key_embedding).transpose(1, 2)  # (N, H, S)
    #     # topic_weight = topic_weight1 + topic_weight2
    #     topic_weight = self.topic_layer(key_embedding).transpose(1, 2)
    #     mask = input_feat["mask"].expand(self.head_num, topic_weight.size(0), -1).transpose(0, 1) == 0
    #     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
    #     topic_vec = self.final(torch.matmul(topic_weight, topic_vec))  # (N, H, E)
    #     return topic_vec, topic_weight
    #
    # def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    #     input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
    #     embedding, topic_weight = self.extract_topic(input_feat)
    #     doc_embedding, doc_topic = self.projection(embedding)  # (N, E), (N, H) (32,768), (32,8,1)
    #     output = self.classify_layer(doc_embedding, topic_weight, return_attention)
    #     if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
    #         entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
    #         output = output + (entropy_sum,)
    #     return output




    # def extract_topic(self, input_feat):
    #     embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
    #     # length = torch.sum(input_feat["mask"], dim=-1)
    #     # y = self.run_gru(embedding, length)
    #     # topic_vec, _ = self.mha1(embedding,embedding,embedding)
    #     key_embedding = self.embedding(input_feat["key_word"], input_feat["mask_key"], inputs_embeds=input_feat["embedding"])[0]
    #     # topic_vec = self.add_norm1(key_embedding, topic_vec)
    #     topic_weight = self.topic_layer(key_embedding).transpose(1, 2)  # (N, H, S)
    #     # expand mask to the same size as topic weights
    #     mask = input_feat["mask"].expand(self.head_num, topic_weight.size(0), -1).transpose(0, 1) == 0
    #     topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
    #     topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
    #     return topic_vec, topic_weight
    #
    # def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
    #     input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)  # (32,8,768), 32,8,100)
    #     embedding, topic_weight = self.extract_topic(input_feat)
    #     doc_embedding, doc_topic = self.projection(embedding)  # (N, E), (N, H) (32,768), (32,8,1)
    #     output = self.classify_layer(doc_embedding, topic_weight, return_attention)
    #     if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
    #         entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
    #         output = output + (entropy_sum,)
    #     return output



# 正常跑的
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         self.projection = AttLayer(self.embed_dim, 128)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat) # (N, H, E), (N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# # 双向 GRU 最高 0.808
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.gru = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.W_k = nn.Linear(2*self.embed_dim, self.head_num, bias=False)
#             self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#             # self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#             self.droputout = nn.Dropout(self.dropout_rate)
#         self.add_norm1 = AddNorm([self.max_length, self.head_num], self.dropout_rate)
#         self.projection = AttLayer(self.embed_dim, 128)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         temp_vec = embedding
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, _ = self.gru(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             temp_vec = self.droputout(self.add_norm1(self.W_q(temp_vec), self.W_k(y)))
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         topic_weight = temp_vec.transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# BATM+GRU-Attention 稳定在0.808
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hidden, weight_out, _ = self.rnn(x, self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(y).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hidden
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         doc_embedding += hn[0].squeeze()
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# 我的模型
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = RNNBase(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             # x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             output, attn_out, weight = self.rnn(embedding, self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(output).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, attn_out
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, attn_out = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         doc_embedding += attn_out
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output





class BiAttentionClassifyModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(BiAttentionClassifyModel, self).__init__(**kwargs)
        self.variant_name = kwargs.pop("variant_name", "base")
        self.with_gru = kwargs.pop("with_gru", None)
        topic_dim = self.head_num * self.head_dim
        # the structure of basic model
        self.final = nn.Linear(self.embed_dim, self.embed_dim)
        if self.variant_name in ["base", "reuse"]:
            self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "topic_embed":
            self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
        else:  # default setting
            self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
        if self.with_gru == "gru":
            # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
            # self.rnn = RNNBase(self.embed_dim, self.embed_dim)
            self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
        elif self.with_gru == "LSTM":
            self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
        elif self.with_gru == "gru_attn":
            self.rnn = RNNBase(self.embed_dim, self.embed_dim)
        self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
        self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
        self.droputout1 = nn.Dropout(0.2)
        self.pooling = nn.MaxPool1d(3,1,1)
        self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)

        self.projection = AttLayer(self.embed_dim, 64)

    def extract_topic(self, input_feat):
        embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
        if self.with_gru:
            length = torch.sum(input_feat["mask"], dim=-1)
            # x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
            output, attn_out, weight = self.rnn(embedding)
            topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(output).transpose(1, 2))) # (N, S, E)
        # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
        # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
        # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
        # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
        mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
        topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
        topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
        return topic_vec, topic_weight, attn_out

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        topic_vec, topic_weight, attn_out = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
        doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
        doc_embedding += attn_out
        output = self.classify_layer(doc_embedding, topic_weight, return_attention)
        if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
            entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
            output = output + (entropy_sum,)
        return output


# elif self.with_gru == "biLSTM":
#                 self.gru = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True, bidirectional=True)


# CNN + BATM
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=True)
#         self.W_k = nn.Linear(2*self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 128)
#
#         # self.num_filters, self.filter_sizes = self.head_num, (3, )
#         # self.conv_layers = nn.ModuleList(
#         #     [nn.Conv2d(1, self.num_filters, (k, self.embed_dim),padding=(1,0)) for k in self.filter_sizes])
#         # self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
#         self.num_filters, self.filter_sizes = self.head_num, (1,)
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,100,256,1)
#         x = x.squeeze(3) # (32,100,256)
#         return x
#
# # 单单 CNN
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         x = embedding.unsqueeze(1) # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)  # (32,(num_filters)H,256)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         topic_weight = x
#         # if self.with_gru:
#         #     length = torch.sum(input_feat["mask"], dim=-1)
#         #     x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#         #     y, _ = self.rnn(x)  # extract interest from history behavior
#         #     y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         #     temp_vec += y
#         topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight

# CNN+rnn融合
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         x = embedding.unsqueeze(1) # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)  # (32,(num_filters)H,256)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         temp_vec = x
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, _ = self.rnn(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# 0.808
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=False)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#         self.num_filters, self.filter_sizes = self.head_num, (1, )
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#         # self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,num_filters,256,1)
#         x = x.squeeze(3) # (32,100,256)  (32,300,256) (32, num_filters, max_length)
#         return x
#
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#
#         x = embedding.unsqueeze(1)  # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         temp_vec = x # (32,H,256)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hn = self.rnn(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_k(y).transpose(1, 2))) # (N, S, E)
#         hn = hn.squeeze(0)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hn
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         doc_embedding += hn
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# 0.810
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(2*self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(2*self.embed_dim, self.embed_dim, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#
#         self.W1 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W2 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W3 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.droputout2 = nn.Dropout(self.dropout_rate)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#         self.num_filters, self.filter_sizes = self.head_num, (1, )
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#         # self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,num_filters,256,1)
#         x = x.squeeze(3) # (32,100,256)  (32,300,256) (32, num_filters, max_length)
#         return x
#
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#
#         x = embedding.unsqueeze(1)  # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         temp_vec = x # (32,H,256)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hn = self.rnn(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(temp_vec, self.W_k(y).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hn
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding = self.droputout2(self.W1(doc_embedding) + self.W2(hn[0].squeeze() + self.W3(hn[-1].squeeze())))
#         doc_embedding += hn[0].squeeze()
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output




# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(2*self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(2*self.embed_dim, self.embed_dim, bias=False)
#         self.W_v = nn.Linear(256, self.head_num, bias=False)
#
#         self.W1 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W2 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W3 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.droputout2 = nn.Dropout(self.dropout_rate)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#         self.add_norm2 = AddNorm(self.embed_dim, self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 128)
#
#         self.num_filters, self.filter_sizes = 256, (1, )
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,num_filters,256,1)
#         x = x.squeeze(3) # (32,100,256)  (32,300,256) (32, num_filters, max_length)
#         return x
#
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#
#         x = embedding.unsqueeze(1)  # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         temp_vec = x # (32,H,256)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hn = self.rnn(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(self.W_v(temp_vec).transpose(1,2), self.W_k(y).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hn
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding = self.droputout2(self.W1(doc_embedding) + self.W2(hn[0].squeeze() + self.W3(hn[-1].squeeze())))
#         hV = hn[0].squeeze()
#         doc_embedding += hV
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(2*self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#
#         self.W1 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W2 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W3 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.droputout2 = nn.Dropout(self.dropout_rate)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#         self.num_filters, self.filter_sizes = self.head_num, (1, )
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#         # self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,num_filters,256,1)
#         x = x.squeeze(3) # (32,100,256)  (32,300,256) (32, num_filters, max_length)
#         return x
#
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hn = self.rnn(x)  # extract interest from history behavior
#             y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(y).transpose(1, 2))) # (N, S, E)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hn
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding = self.droputout2(self.W1(doc_embedding) + self.W2(hn[0].squeeze() + self.W3(hn[-1].squeeze())))
#         doc_embedding += hn[0].squeeze()
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


#
# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#
#         self.W1 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W2 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.W3 = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.droputout2 = nn.Dropout(self.dropout_rate)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#         self.num_filters, self.filter_sizes = self.head_num, (1, )
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
#         # self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)) #(32,num_filters,256,1)
#         x = x.squeeze(3) # (32,100,256)  (32,300,256) (32, num_filters, max_length)
#         return x
#
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         x = embedding.unsqueeze(1)  # （32，1，256，300）
#         x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
#         x = nn.Dropout(self.dropout_rate)(x)  # (32,(num_filters)H,256)
#         temp_vec = x # (32,H,256)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hidden,weight_out, _ = self.rnn(x, self.max_length)  # extract interest from history behavior
#             # y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout1(self.add_norm1(self.W_k(embedding).transpose(1, 2), self.W_q(y).transpose(1, 2))) # (N, S, H)
#         # topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         topic_weight = temp_vec # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         # topic_weight = torch.softmax(topic_weight, dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, y))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, weight_out
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, weight_out = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H, 1)
#         # doc_embedding = self.droputout2(self.W1(doc_embedding) + self.W2(hn[0].squeeze() + self.W3(hn[-1].squeeze())))
#         doc_embedding += weight_out
#         output = self.classify_layer(weight_out, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output


# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         temp_vec = embedding
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hidden, weight_out, _ = self.rnn(x, self.max_length)
#             temp_vec += y
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(y).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(temp_vec).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, temp_vec))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hidden
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         doc_embedding += hn[0].squeeze()
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output




# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             y, hidden, weight_out, _ = self.rnn(x, self.max_length)
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(y).transpose(1, 2))) # (N, S, E)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, hidden
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, hn = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding += hn[0].squeeze()
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output




# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             # x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             output, attn_out, weight  = self.rnn(embedding, self.max_length) # [32,300],  [32,256]
#             topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(output).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(output).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return topic_vec, topic_weight, attn_out
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight, attn_out = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding += hn[0].squeeze()
#         doc_embedding += attn_out
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output



# class BiAttentionClassifyModel(BaseClassifyModel):
#     def __init__(self, **kwargs):
#         super(BiAttentionClassifyModel, self).__init__(**kwargs)
#         self.variant_name = kwargs.pop("variant_name", "base")
#         self.with_gru = kwargs.pop("with_gru", None)
#         topic_dim = self.head_num * self.head_dim
#         # the structure of basic model
#         self.final = nn.Linear(self.embed_dim, self.embed_dim)
#         if self.variant_name in ["base", "reuse"]:
#             self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
#                                              nn.Linear(topic_dim, self.head_num))
#         elif self.variant_name == "topic_embed":
#             self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
#         else:  # default setting
#             self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
#         if self.with_gru == "gru":
#             # self.rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True,bidirectional=True)
#             self.rnn = GRUWithAttention(self.embed_dim, self.embed_dim)
#         elif self.with_gru == "LSTM":
#             self.rnn = nn.LSTM(self.embed_dim, self.embed_dim, 1, batch_first=True, bidirectional=False)
#         self.W_k = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_q = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.W_v = nn.Linear(self.embed_dim, self.head_num, bias=False)
#         self.droputout1 = nn.Dropout(0.2)
#         self.pooling = nn.MaxPool1d(3,1,1)
#         self.add_norm1 = AddNorm([self.head_num,self.max_length], self.dropout_rate)
#
#         self.projection = AttLayer(self.embed_dim, 64)
#
#     def extract_topic(self, input_feat):
#         embedding = self.embedding_layer(input_feat)  # (N, S) -> (N, S, E)
#         if self.with_gru:
#             length = torch.sum(input_feat["mask"], dim=-1)
#             # x = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
#             output, attn_out, weight  = self.rnn(embedding, self.max_length) # [32,300],  [32,256]
#             # topic_weight = self.droputout1(self.add_norm1(self.W_q(embedding).transpose(1,2), self.W_k(output).transpose(1, 2))) # (N, S, E)
#         # topic_weight = self.droputout(self.add_norm1(temp_vec, self.W_v(embedding).transpose(1, 2),self.W_k(y).transpose(1,2)))
#         # topic_weight = self.droputout(self.add_norm1(self.W_q(embedding).transpose(1, 2), topic_weight)) # (N, S, H)
#         # topic_weight = self.topic_layer(output).transpose(1, 2)  # (N, S, E) --> (N, H, S)
#         # topic_weight = temp_vec.transpose(1, 2) # (N, S, E) --> (N, H, S)
#         # mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
#         # topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
#         # topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, S) * (N, S, E) -> (N, H, E) 得到的是每个主题的主题向量
#         return attn_out, weight
#
#     def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
#         input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
#         topic_vec, topic_weight = self.extract_topic(input_feat) # (N, H, E),(N, H, S)
#         doc_embedding, doc_topic = self.projection(topic_vec)  # (N, H, E) --> (N, E), (N, H)
#         # doc_embedding += hn[0].squeeze()
#         # doc_embedding += attn_out
#         output = self.classify_layer(doc_embedding, topic_weight, return_attention)
#         if self.entropy_constraint or self.calculate_entropy:  # (32,15),(32,8,100)
#             entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
#             output = output + (entropy_sum,)
#         return output
