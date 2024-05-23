import pandas as pd
import random
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from utils.preprocess_utils import clean_text, text2index
from utils.general_utils import read_json, write_json


def clean_df(data_df):
    # 这行代码的作用是从数据表 data_df 中删除那些在"标题" (title) 和 "正文" (body) 字段中同时为空的行。参数 inplace=True 表示这一操作直接就地对 data_df 进行修改。
    data_df.dropna(subset=["title", "body"], inplace=True, how="all")
    # 这行代码的作用是将 data_df 中的空值（NA 或 NaN）替换为字符串 "empty"。参数 inplace=True 表示这一操作直接在 data_df 上进行修改。
    data_df.fillna("empty", inplace=True)
    # 这行代码使用一个匿名函数（lambda 函数）去处理 data_df 中的 title 列。函数 clean_text(s) 应该是一个对字符串进行清洗的函数，即对每一篇文章的标题进行清洗。
    data_df["title"] = data_df.title.apply(lambda s: clean_text(s))
    data_df["body"] = data_df.body.apply(lambda s: clean_text(s))
    return data_df


'''
这段代码是对Pandas DataFrame的一组操作，主要用于创建一个随机筛选的验证集。以下是对每行代码的解析：
df是一个Pandas的DataFrame对象，你可以将它视为一个二维的数据表格。
indices = df.index.values：这行代码获取df的索引，也就是行号，并赋值给indices变量。例如，如果df有10行，indices就是一个包含0到9的数组。
random.Random(42).shuffle(indices)：这行代码使用shuffle方法对indices数组进行随机排序。注意这里的42是随机数生成器的种子，保证了每次运行这段代码，得到的随机排序都是一样的。
split_len = round(split * len(df))：这行代码计算验证集的长度。split是一个介于0和1之间的浮点数，代表验证集在所有数据中的占比。len(df)则是df的行数。所以split * len(df)就是我们期望的验证集大小，然后通过round函数进行四舍五入。
df.loc[indices[:split_len], "split"] = "valid"：这行代码实际执行了数据集的分割。它选取了df中索引在乱序indices数组前split_len部分的行，也就是随机选取的split_len数量的数据，并在这些行下新增了一列名为"split"的属性，标记为"valid"。
'''
# 这个函数就是划分test,train,valid数据集的
def split_df(df, split=0.1, split_test=False):
    indices = df.index.values
    random.Random(42).shuffle(indices)
    split_len = round(split * len(df))
    df.loc[indices[:split_len], "split"] = "valid"
    if split_test:
        df.loc[indices[split_len:split_len*2], "split"] = "test"
        df.loc[indices[split_len*2:], "split"] = "train"
    else:
        df.loc[indices[split_len:], "split"] = "train"
    return df


def load_set_by_type(dataset, set_type: str) -> pd.DataFrame:
    df = {k: [] for k in ["data", "category"]}
    for text, label in zip(dataset[set_type]["text"], dataset[set_type]["label"]):
        for c, v in zip(["data", "category"], [text, label]):
            df[c].append(v)
    df["split"] = set_type
    return pd.DataFrame(df)


# 单单使用关键词的写法
# def load_dataset_df(dataset_name, data_path):
#     if dataset_name in ["MIND15", "News26", "New_MIND15"]:
#         # 这里是直接读取本地的数据，完蛋
#         df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
#         df["data"] = df.title + "\n" + df.body
#     elif dataset_name in ["MIND15-Roberta-3"]:
#         df = pd.read_csv(data_path, encoding="utf-8")
#         df["data"] = df.text
#         labels = df["label"].values.tolist()
#         label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#         return df, label_dict
#     elif dataset_name in ["ag_news", "yelp_review_full", "imdb"]:
#         # load corresponding dataset from datasets library，使用 NewsDataLoader 类里面的 load_dataset
#         dataset = load_dataset(dataset_name)
#         train_set, test_set = split_df(load_set_by_type(dataset, "train")), load_set_by_type(dataset, "test")
#         df = train_set.append(test_set)
#     else:
#         raise ValueError("dataset name should be in one of MIND15, IMDB, News26, and ag_news...")
#     labels = df["category"].values.tolist()
#     label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#     return df, label_dict

# 同时使用关键词和原始文本的写法
# def load_dataset_df(dataset_name, data_path):
#     if dataset_name in ["MIND15", "News26"]:
#         # 这里是直接读取本地的数据，完蛋
#         df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
#         df["data"] = df.title + "\n" + df.body
#         df1 = pd.read_csv("D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\data\\MIND15-Roberta-3.csv", encoding="utf-8")
#         df["key"] = df1.text
#         df.index=list(range(len(df)))
#     elif dataset_name in ["ag_news", "yelp_review_full", "imdb"]:
#         # load corresponding dataset from datasets library，使用 NewsDataLoader 类里面的 load_dataset
#         dataset = load_dataset(dataset_name)
#         train_set, test_set = split_df(load_set_by_type(dataset, "train")), load_set_by_type(dataset, "test")
#         df = train_set.append(test_set)
#     else:
#         raise ValueError("dataset name should be in one of MIND15, IMDB, News26, and ag_news...")
#     labels = df["category"].values.tolist()
#     label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
#     return df, label_dict

# # CNEWS的写法
def load_dataset_df(dataset_name, data_path):
    if dataset_name in ["MIND15", "News26"]:
            df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
            df["data"] = df.title + "\n" + df.body
    elif dataset_name in ["cnews", "toutiao_news"]:
        # 这里是直接读取本地的数据，完蛋
        df = pd.read_csv(data_path, encoding="utf-8")
    elif dataset_name in ["ag_news", "yelp_review_full", "imdb"]:
        # load corresponding dataset from datasets library，使用 NewsDataLoader 类里面的 load_dataset
        dataset = load_dataset(dataset_name)
        train_set, test_set = split_df(load_set_by_type(dataset, "train")), load_set_by_type(dataset, "test")
        df = train_set.append(test_set)
    else:
        raise ValueError("dataset name should be in one of MIND15, IMDB, News26, and ag_news...")
    labels = df["label"].values.tolist()
    label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
    return df, label_dict


def load_word_dict(data_root, dataset_name, process_method, **kwargs):
    embed_method = kwargs.get("embed_method", "use_all")
    wd_path = Path(data_root) / "utils" / "word_dict" / f"{dataset_name}_{process_method}_{embed_method}.json"
    if os.path.exists(wd_path):
        word_dict = read_json(wd_path)
    else:
        word_dict = {}
        data_path = kwargs.get("data_path", Path(data_root) / "data" / f"{dataset_name}.csv")
        df = kwargs.get("df", load_dataset_df(dataset_name, data_path)[0])
        df.data.apply(lambda s: text2index(s, word_dict, process_method, False))
        os.makedirs(wd_path.parent, exist_ok=True)
        write_json(word_dict, wd_path)
    return word_dict


def load_glove_embedding(glove_path=None):
    if not glove_path:
        glove_path = "D:\\AI\\Graduation_Project\\model\\BATM\\dataset\\glove\\glove.840B.300d.txt"
        # 这里要使用的是相对路径
        # glove_path = '/kaggle/input/d/supomaker/mind15-and-news26-and-glove/dataset/glove/glove.840B.300d.txt'
    glove = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
    return {key: val.values for key, val in glove.T.items()}

'''
这个函数其实就是加载glove embedding, 并使用glove的embedding将词典word_dict中出现过的单词重新赋值给new_wd,就是使用跟 glove embedding一样的词表索引
'''
def load_embeddings(data_root, dataset_name, process_method, word_dict, glove_path=None, embed_method="use_all"):
    embed_path = Path(data_root) / "utils" / "embed_dict" / f"{dataset_name}_{process_method}_{embed_method}.npy"
    wd_path = Path(data_root) / "utils" / "word_dict" / f"{dataset_name}_{process_method}_{embed_method}.json"
    if os.path.exists(embed_path):
        embeddings = np.load(embed_path.__str__())
        word_dict = read_json(wd_path)
    else:
        new_wd = {"[UNK]": 0}
        # 这里加载已经处理好的embedding_dict
        embedding_dict = load_glove_embedding(glove_path)
        embeddings, exclude_words = [np.zeros(300)], []
        for i, w in enumerate(word_dict.keys()):
            if w in embedding_dict:
                embeddings.append(embedding_dict[w])
                new_wd[w] = len(new_wd)
            else:
                exclude_words.append(w)
        if embed_method == "use_all":
            mean, std = np.mean(embeddings), np.std(embeddings)
            # append random embedding
            for i, w in enumerate(exclude_words):
                new_wd[w] = len(new_wd)
                embeddings.append(np.random.normal(loc=mean, scale=std, size=300))
        os.makedirs(embed_path.parent, exist_ok=True)
        np.save(embed_path.__str__(), np.array(embeddings))
        word_dict = new_wd
        write_json(word_dict, wd_path)
    return np.array(embeddings), word_dict
