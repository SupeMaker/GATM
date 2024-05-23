import os
from pathlib import Path
from torch.utils.data import DataLoader
from utils import load_dataset_df, load_word_dict, load_embeddings
from base.base_dataset import BaseDataset, BaseDatasetBert


# class NewsDataLoader:
#     def load_dataset(self, df):
#         pretrained_models = ["distilbert-base-uncased", "bert-base-uncased", "xlnet-base-cased", "roberta-base",
#                              "longformer-base-4096", "transfo-xl-wt103", "roberta-wwm"]
#         if self.embedding_type in pretrained_models:
#             # df["data"] = df.title + "\n" + df.body
#             # 这里是根据 embedding_type 得到对应模型的dataset,dataset中含有tokenizer最为关键
#             dataset = BaseDatasetBert(texts=df["data"].values.tolist(), labels=df["category"].values.tolist(),
#                                       label_dict=self.label_dict, max_length=self.max_length,
#                                       embedding_type=self.embedding_type,is_local=True)
#             if self.embedding_type == "transfo-xl-wt103":
#                 # 根据给定的预训练模型类型(embedding_type)生成相应的分词器(tokenizer)并获取其词汇表中每个符号的索引。
#                 self.word_dict = dataset.tokenizer.sym2idx
#             else:
#                 self.word_dict = dataset.tokenizer.vocab
#         elif self.embedding_type in ["glove", "init"]:
#             # if we use glove embedding, then we ignore the unknown words
#             # dataset = BaseDatasetBert(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
#             #                       self.max_length, self.word_dict, self.method)
#             dataset = BaseDataset(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
#                               self.max_length, self.word_dict, self.method)
#         else:
#             raise ValueError(f"Embedding type should be one of {','.join(pretrained_models)} or glove and init")
#         return dataset
#
#     def __init__(self, batch_size=32, shuffle=True, num_workers=1, max_length=128, name="MIND15/keep", **kwargs):
#         self.set_name, self.method = name.split("/")[0], name.split("/")[1]
#         # print("self.set_name, self.method: ", self.set_name, self.method) #  News26 keep_all
#         # kwargs.get("embedding_type", "glove") 尝试从kwargs中获取"embedding_type"的值。如果字典中有"embedding_type"这个键，那么就返回其对应的值；如果字典中没有"embedding_type"这个键，那么方法就会返回默认值"glove"。
#         self.max_length, self.embedding_type = max_length, kwargs.get("embedding_type", "glove")
#         # self.data_root = kwargs.get("data_root", "../../dataset")
#
#         # self.data_root = "../../dataset"
#         self.data_root = "./dataset"
#         # self.data_root = "/kaggle/input/d/supomaker/mind15-and-news26-and-glove/dataset"
#         # print("self.data_root: ", self.data_root) # self.data_root:  .\dataset
#         data_path = Path(self.data_root) / "data" / f"{self.set_name}.csv"
#         # print("data_path: ", data_path) # data_path:  dataset\data\News26.csv
#         # 加载数据
#         df, self.label_dict = load_dataset_df(self.set_name, data_path)
#         # 在这里划分数据集
#         train_set, valid_set, test_set = df["split"] == "train", df["split"] == "valid", df["split"] == "test"
#         if self.embedding_type in ["glove", "init"]:
#             # setup word dictionary for glove or init embedding
#             self.word_dict = load_word_dict(self.data_root, self.set_name, self.method, df=df)
#         if self.embedding_type == "glove":
#             # 这里加载 glove的embedding表示
#             self.embeds, self.word_dict = load_embeddings(self.data_root, self.set_name, self.method, self.word_dict,
#                                                           embed_method=kwargs.get("embed_method", "use_all"))
#         self.init_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
#         # initialize train loader
#         self.train_loader = DataLoader(self.load_dataset(df[train_set]), **self.init_params)
#         # initialize validation loader
#         self.valid_loader = DataLoader(self.load_dataset(df[valid_set]), **self.init_params)
#         # initialize test loader
#         self.test_loader = DataLoader(self.load_dataset(df[test_set]), **self.init_params)
#

# 使用关键词的写法
# class NewsDataLoader:
#     def load_dataset(self, df):
#         pretrained_models = ["distilbert-base-uncased", "bert-base-uncased", "xlnet-base-cased", "roberta-base",
#                              "longformer-base-4096", "transfo-xl-wt103"]
#         if self.embedding_type in pretrained_models:
#             # df["data"] = df.title + "\n" + df.body
#             # 这里是根据 embedding_type 得到对应模型的dataset,dataset中含有tokenizer最为关键
#             dataset = BaseDatasetBert(texts=df["data"].values.tolist(), key_word=df["key"].values.tolist(), labels=df["category"].values.tolist(),
#                                       label_dict=self.label_dict, max_length=self.max_length,
#                                       embedding_type=self.embedding_type,is_local=True)
#             if self.embedding_type == "transfo-xl-wt103":
#                 # 根据给定的预训练模型类型(embedding_type)生成相应的分词器(tokenizer)并获取其词汇表中每个符号的索引。
#                 self.word_dict = dataset.tokenizer.sym2idx
#             else:
#                 self.word_dict = dataset.tokenizer.vocab
#         elif self.embedding_type in ["glove", "init"]:
#             # if we use glove embedding, then we ignore the unknown words
#             # dataset = BaseDatasetBert(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
#             #                       self.max_length, self.word_dict, self.method)
#             dataset = BaseDataset(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
#                               self.max_length, self.word_dict, self.method)
#         else:
#             raise ValueError(f"Embedding type should be one of {','.join(pretrained_models)} or glove and init")
#         return dataset
#
#     def __init__(self, batch_size=32, shuffle=True, num_workers=1, max_length=128, name="MIND15/keep", **kwargs):
#         self.set_name, self.method = name.split("/")[0], name.split("/")[1]
#         # print("self.set_name, self.method: ", self.set_name, self.method) #  News26 keep_all
#         # kwargs.get("embedding_type", "glove") 尝试从kwargs中获取"embedding_type"的值。如果字典中有"embedding_type"这个键，那么就返回其对应的值；如果字典中没有"embedding_type"这个键，那么方法就会返回默认值"glove"。
#         self.max_length, self.embedding_type = max_length, kwargs.get("embedding_type", "glove")
#         # self.data_root = kwargs.get("data_root", "../../dataset")
#
#         # self.data_root = "../../dataset"
#         self.data_root = "./dataset"
#         # self.data_root = "/kaggle/input/d/supomaker/mind15-and-news26-and-glove/dataset"
#         # print("self.data_root: ", self.data_root) # self.data_root:  .\dataset
#         data_path = Path(self.data_root) / "data" / f"{self.set_name}.csv"
#         # print("data_path: ", data_path) # data_path:  dataset\data\News26.csv
#         # 加载数据
#         df, self.label_dict = load_dataset_df(self.set_name, data_path)
#         # 在这里划分数据集
#         train_set, valid_set, test_set = df["split"] == "train", df["split"] == "valid", df["split"] == "test"
#         if self.embedding_type in ["glove", "init"]:
#             # setup word dictionary for glove or init embedding
#             self.word_dict = load_word_dict(self.data_root, self.set_name, self.method, df=df)
#         if self.embedding_type == "glove":
#             # 这里加载 glove的embedding表示
#             self.embeds, self.word_dict = load_embeddings(self.data_root, self.set_name, self.method, self.word_dict,
#                                                           embed_method=kwargs.get("embed_method", "use_all"))
#         self.init_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
#         # initialize train loader
#         self.train_loader = DataLoader(self.load_dataset(df[train_set]), **self.init_params)
#         # initialize validation loader
#         self.valid_loader = DataLoader(self.load_dataset(df[valid_set]), **self.init_params)
#         # initialize test loader
#         self.test_loader = DataLoader(self.load_dataset(df[test_set]), **self.init_params)


# # 使用cnews
class NewsDataLoader:
    def load_dataset(self, df):
        pretrained_models = ["distilbert-base-uncased", "bert-base-uncased", "xlnet-base-cased", "roberta-base",
                             "longformer-base-4096", "transfo-xl-wt103", "bert-base-chinese", "roberta-wwm"]
        if self.embedding_type in pretrained_models:
            # df["data"] = df.title + "\n" + df.body
            # 这里是根据 embedding_type 得到对应模型的dataset,dataset中含有tokenizer最为关键
            dataset = BaseDatasetBert(texts=df["body"].values.tolist(), labels=df["label"].values.tolist(),
                                      label_dict=self.label_dict, max_length=self.max_length,
                                      embedding_type=self.embedding_type,is_local=True)
            if self.embedding_type == "transfo-xl-wt103":
                # 根据给定的预训练模型类型(embedding_type)生成相应的分词器(tokenizer)并获取其词汇表中每个符号的索引。
                self.word_dict = dataset.tokenizer.sym2idx
            else:
                self.word_dict = dataset.tokenizer.vocab
        elif self.embedding_type in ["glove", "init"]:
            # if we use glove embedding, then we ignore the unknown words
            # dataset = BaseDatasetBert(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
            #                       self.max_length, self.word_dict, self.method)
            dataset = BaseDataset(df["body"].values.tolist(), df["label"].values.tolist(), self.label_dict,
                              self.max_length, self.word_dict, self.method)
        else:
            raise ValueError(f"Embedding type should be one of {','.join(pretrained_models)} or glove and init")
        return dataset

    def __init__(self, batch_size=32, shuffle=True, num_workers=1, max_length=128, name="MIND15/keep", **kwargs):
        self.set_name, self.method = name.split("/")[0], name.split("/")[1]
        # print("self.set_name, self.method: ", self.set_name, self.method) #  News26 keep_all
        # kwargs.get("embedding_type", "glove") 尝试从kwargs中获取"embedding_type"的值。如果字典中有"embedding_type"这个键，那么就返回其对应的值；如果字典中没有"embedding_type"这个键，那么方法就会返回默认值"glove"。
        self.max_length, self.embedding_type = max_length, kwargs.get("embedding_type", "glove")
        # self.data_root = kwargs.get("data_root", "../../dataset")

        # self.data_root = "../../dataset"
        self.data_root = "./dataset"
        # self.data_root = "/kaggle/input/d/supomaker/mind15-and-news26-and-glove/dataset"
        # print("self.data_root: ", self.data_root) # self.data_root:  .\dataset
        data_path = Path(self.data_root) / "data" / f"{self.set_name}.csv"
        # print("data_path: ", data_path) # data_path:  dataset\data\News26.csv
        # 加载数据
        df, self.label_dict = load_dataset_df(self.set_name, data_path)
        # 在这里划分数据集
        train_set, valid_set, test_set = df["split"] == "train", df["split"] == "valid", df["split"] == "test"
        if self.embedding_type in ["glove", "init"]:
            # setup word dictionary for glove or init embedding
            self.word_dict = load_word_dict(self.data_root, self.set_name, self.method, df=df)
        if self.embedding_type == "glove":
            # 这里加载 glove的embedding表示
            self.embeds, self.word_dict = load_embeddings(self.data_root, self.set_name, self.method, self.word_dict,
                                                          embed_method=kwargs.get("embed_method", "use_all"))
        self.init_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        # initialize train loader
        self.train_loader = DataLoader(self.load_dataset(df[train_set]), **self.init_params)
        # initialize validation loader
        self.valid_loader = DataLoader(self.load_dataset(df[valid_set]), **self.init_params)
        # initialize test loader
        self.test_loader = DataLoader(self.load_dataset(df[test_set]), **self.init_params)