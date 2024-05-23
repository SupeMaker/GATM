import logging
from typing import List, Mapping

import numpy as np
import  re
import torch
from keybert import KeyBERT
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import text2index


class BaseDataset(Dataset):
    def __init__(self, texts, labels, label_dict, max_length, word_dict, process_method="keep_all"):
        super().__init__()
        self.texts, self.labels, self.label_dict, self.max_length = texts, labels, label_dict, max_length
        self.word_dict = word_dict
        self.process_method = process_method

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

    def __getitem__(self, i):
        data = text2index(self.texts[i], self.word_dict, self.process_method, True)[:self.max_length]
        data.extend([0 for _ in range(max(0, self.max_length - len(data)))])
        data = torch.tensor(data, dtype=torch.long)
        label = torch.tensor(self.label_dict.get(self.labels[i], -1), dtype=torch.long).squeeze(0)
        mask = torch.tensor(np.where(data == 0, 0, 1), dtype=torch.long)
        return {"data": data, "label": label, "mask": mask}

    def __len__(self):
        return len(self.labels)



# class BaseDatasetBert(Dataset):
#     def __init__(self, texts: List[str], labels: List[str] = None, label_dict: Mapping[str, int] = None,
#                  max_length: int = 512, embedding_type: str = 'distilbert-base-uncased', is_local=False):
#
#         self.texts = texts
#         self.labels = labels
#         self.label_dict = label_dict
#         self.max_length = max_length
#
#         if self.label_dict is None and labels is not None:
#             self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#         if is_local:
#             model_root = "D:\\AI\\model\\" + embedding_type
#         else:
#             model_root = embedding_type
#         # 这里下载 tokenizer
#         # Please provide either the path to a local folder or the repo_id of a model on the Hub.
#         self.tokenizer = AutoTokenizer.from_pretrained(model_root)
#         self.keybert_model = KeyBERT(model='D:\\AI\\model\\roberta-base')
#         logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)
#         # self.sep_vid = self.tokenizer.sep_token_id
#         # self.cls_vid = self.tokenizer.cls_token_id
#         if embedding_type == "transfo-xl-wt103":
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.pad_vid = self.tokenizer.pad_token_id
#         else:
#             self.pad_vid = self.tokenizer.pad_token_id
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
#
#         x = self.texts[index]
#         # key_word = self.keybert_model.extract_keywords(x, keyphrase_ngram_range=(1, 2), stop_words="english")
#         # key_word = self.concat_words(key_word)
#         x_encoded = self.tokenizer.encode(x, add_special_tokens=True, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze(0)
#
#         # 这里是得到等长的embedding，会进行填充，获得处理过的X和mask
#         true_seq_length = x_encoded.size(0)
#         pad_size = self.max_length - true_seq_length
#         pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
#         x_tensor = torch.cat((x_encoded, pad_ids))
#
#         mask = torch.ones_like(x_encoded, dtype=torch.int8)
#         mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
#         mask = torch.cat((mask, mask_pad))
#         # output_dict = {"data": x_tensor, 'mask': mask, "text": x}
#         output_dict = {"data": x_tensor, 'mask': mask}
#         if self.labels is not None:
#             y = self.labels[index]
#             y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
#             output_dict["label"] = y_encoded
#
#         return output_dict
#
#     def concat_words(self, keywords):
#         w = [x[0] for x in keywords]
#         return " ".join(w)


#
# class BaseDatasetBert(Dataset):
#     def __init__(self, texts: List[str], key_word:List[str], labels: List[str] = None, label_dict: Mapping[str, int] = None,
#                  max_length: int = 512, embedding_type: str = 'distilbert-base-uncased', is_local=False):
#
#         self.texts = texts
#         self.key_word = key_word
#         self.labels = labels
#         self.label_dict = label_dict
#         self.max_length = max_length
#
#         if self.label_dict is None and labels is not None:
#             self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#         if is_local:
#             model_root = "D:\\AI\\model\\" + embedding_type
#         else:
#             model_root = embedding_type
#         # 这里下载 tokenizer
#         # Please provide either the path to a local folder or the repo_id of a model on the Hub.
#         self.tokenizer = AutoTokenizer.from_pretrained(model_root)
#         logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)
#         # self.sep_vid = self.tokenizer.sep_token_id
#         # self.cls_vid = self.tokenizer.cls_token_id
#         if embedding_type == "transfo-xl-wt103":
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.pad_vid = self.tokenizer.pad_token_id
#         else:
#             self.pad_vid = self.tokenizer.pad_token_id
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
#
#         origin_text = self.texts[index]
#         key_text = self.key_word[index]
#         origin_text_encoded = self.tokenizer.encode(origin_text, add_special_tokens=True, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze(0)
#         key_text_encoded = self.tokenizer.encode(key_text, add_special_tokens=True, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze(0)
#
#         # 这里是得到等长的embedding，会进行填充，获得处理过的X和mask
#         true_seq_length = origin_text_encoded.size(0)
#         pad_size = self.max_length - true_seq_length
#         pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
#         origin_text_tensor = torch.cat((origin_text_encoded, pad_ids))
#
#         key_true_seq_length = key_text_encoded.size(0)
#         pad_size = self.max_length - key_true_seq_length
#         pad_ids_key = torch.Tensor([self.pad_vid] * pad_size).long()
#         key_text_encoded_tensor = torch.cat((key_text_encoded, pad_ids_key))
#
#         mask = torch.ones(true_seq_length, dtype=torch.int8)
#         mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
#         mask = torch.cat((mask, mask_pad))
#
#         mask_key = torch.ones(key_true_seq_length, dtype=torch.int8)
#         mask_key_pad = torch.zeros_like(pad_ids_key, dtype=torch.int8)
#         mask_key = torch.cat((mask_key, mask_key_pad))
#
#         output_dict = {"data": origin_text_tensor, 'mask': mask, "key_word":key_text_encoded_tensor, "mask_key": mask_key}
#         if self.labels is not None:
#             y = self.labels[index]
#             y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
#             output_dict["label"] = y_encoded
#
#         return output_dict


class BaseDatasetBert(Dataset):
    def __init__(self, texts: List[str], labels: List[str] = None, label_dict: Mapping[str, int] = None,
                 max_length: int = 512, embedding_type: str = 'distilbert-base-uncased', is_local=False):

        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_length = max_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
        if is_local:
            model_root = "D:\\AI\\model\\" + embedding_type
        else:
            model_root = embedding_type
        # 这里下载 tokenizer
        # Please provide either the path to a local folder or the repo_id of a model on the Hub.
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)
        if embedding_type == "transfo-xl-wt103":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_vid = self.tokenizer.pad_token_id
        else:
            self.pad_vid = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        x = self.texts[index]
        x_encoded = self.tokenizer.encode(x, add_special_tokens=True, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze(0)

        # 这里是得到等长的embedding，会进行填充，获得处理过的X和mask
        true_seq_length = x_encoded.size(0)
        pad_size = self.max_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))
        output_dict = {"data": x_tensor, 'mask': mask}
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["label"] = y_encoded

        return output_dict
