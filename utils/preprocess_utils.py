import re
import string

from nltk.corpus import stopwords as stop_words
from torchtext.data import get_tokenizer


def text2index(text, word_dict, method="keep", ignore=True):
    return word2index(word_dict, tokenize(text, method), ignore)

'''
这个 clean_text 函数的作用是清洗文本，它主要做的是移除文本中的标点符号以及数字。让我们来具体看一下这个函数是如何工作的：
string.punctuation + "0123456789"：此处创建了一个字符串规则（rule），这个字符串包含了所有的标点符号以及数字0到9。
re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)：这个是 Python 的正则表达式 re.sub() 方法，它用于替换字符串中的匹配项。这里它将所有不在 rule 中，也不是英文字母和空格的字符替换为一个空格。rf'([^{rule}a-zA-Z ])' 表示匹配所有不在 rule、不是英文小写字母、不是英文大写字母、也不是空格的字符。
'''
def clean_text(text):
    rule = string.punctuation + "0123456789"
    return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)


def aggressive_process(text):
    stopwords = set(stop_words.words("english"))
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
    text = [w for w in text.split() if len(w) > 0 and w not in stopwords]
    return text


def tokenize(text, method="keep_all"):
    tokens = []
    text = clean_text(text)
    rule = string.punctuation + "0123456789"
    tokenizer = get_tokenizer('basic_english')
    if method == "keep_all":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" \1 ", text.lower()))
    elif method == "aggressive":
        tokens = aggressive_process(text)
    elif method == "alphabet_only":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" ", text.lower()))
    return tokens


def word2index(word_dict, sent, ignore=True):
    word_index = []
    for word in sent:
        if ignore:
            index = word_dict[word] if word in word_dict else 0
        else:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            index = word_dict[word]
        word_index.append(index)
    return word_index
