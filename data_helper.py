# -*- coding: utf-8 -*-



import re
import torch
from torch.utils.data import Dataset
from collections import Counter
from config import base_config
from utils import dump_json, load_json
from transformers import BertTokenizer



# 词典大小
VOCAB_SIZE = base_config["vocab_size"]

# 数据路径
TRAIN_DATA_PATH = base_config["train_data_path"]
TEST_DATA_PATH = base_config["test_data_path"]
VAL_DATA_PATH = base_config["val_data_path"]
WORD2IDX_PATH = base_config["word2idx_path"]
LABEL2IDX_PATH = base_config["label2idx_path"]


# 数据预处理，主要是生成字索引以及标签索引
def pre_process_data():
    words = []
    labels = []
    with open(TRAIN_DATA_PATH) as f:
        for line in f:
            lis = line.split("\t")
            label, text =lis[0], lis[1]
            words.extend(list(text))
            if label not in labels:
                labels.append(label)
    word_count = Counter(words).most_common(VOCAB_SIZE-2)
    word_dic = ["<PAD>", "<UNK>"]
    for w, c in word_count:
        word_dic.append(w)
    word2idx = {w:idx for idx, w in enumerate(word_dic)}
    label2idx = {l:idx for idx, l in enumerate(labels)}

    dump_json(word2idx, WORD2IDX_PATH)
    dump_json(label2idx, LABEL2IDX_PATH)


def load_init_data():
    # 加载word2idx, label2idx
    word2idx = load_json(WORD2IDX_PATH)
    label2idx = load_json(LABEL2IDX_PATH)
    max_doc_len = base_config["max_doc_len"]

    # 下面两项主要用于DPCNN
    max_sent_len = base_config["max_sent_len"]
    max_words_len = base_config["max_words_len"]
    return word2idx, label2idx, max_doc_len, max_sent_len, max_words_len


class CnnRnnDataSet(Dataset):
    def __init__(self, data_path):
        word2idx, label2idx, max_doc_len, max_sent_len, max_words_len = load_init_data()
        labels = []
        texts = []
        with open(data_path) as f:
            for line in f:
                l, t = line.strip().split("\t")
                labels.append(l)
                texts.append(t)
        self.text_ids, self.label_ids = self.padding_data(texts, labels, word2idx, label2idx, max_doc_len)

    def padding_data(self, texts, labels, word2idx, label2idx, max_doc_len):
        text_ids = []
        label_ids = []
        pad_idx = word2idx["<PAD>"]
        unk_idx = word2idx["<UNK>"]
        for t, l in zip(texts, labels):
            tmp = [pad_idx] * max_doc_len
            ids = [word2idx.get(w, unk_idx) for w in t][:max_doc_len]
            tmp[:len(ids)] = ids
            text_ids.append(tmp)
            label_ids.append(label2idx[l])
        return torch.tensor(text_ids), torch.tensor(label_ids)


    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, item):
        return self.text_ids[item], self.label_ids[item]


class BertDataSet(Dataset):
    def __init__(self, data_path):
        self.tokenizer = BertTokenizer.from_pretrained(base_config["pre_bert_model"])
        word2idx, label2idx, self.max_doc_len, max_sent_len, max_words_len = load_init_data()
        labels = []
        texts = []
        with open(data_path) as f:
            for line in f:
                l, t = line.strip().split("\t")
                labels.append(l)
                texts.append(list(t))

        self.input_ids, self.token_type_ids, self.attention_mask = self.encode_fn(texts)
        self.label_ids = torch.tensor([label2idx[l] for l in labels])



    def encode_fn(self, text_list):
        """将输入句子编码成BERT需要格式"""
        tokenizers = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors='pt',  # 返回的类型为pytorch tensor
            is_split_into_words=True
        )
        input_ids = tokenizers['input_ids']
        token_type_ids = tokenizers['token_type_ids']
        attention_mask = tokenizers['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, item):
        return self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], self.label_ids[item]


class HanDataSet(Dataset):
    def __init__(self, data_path):
        self.word2idx, self.label2idx, self.max_doc_len, self.max_sent_len, self.max_words_len = load_init_data()
        texts = []
        labels = []
        with open(data_path) as f:
            for line in f:
                l, t = line.strip().split("\t")
                labels.append(l)
                texts.append(t)
        self.text_ids, self.sent_len_seq, self.token_len_seq, self.label_ids = self.padding_data(
            texts, labels
        )


    def split_sents(self, text):
        """将文本分句"""
        sents = re.split("([。！？])", text)
        sents.append("")
        res = []
        for t, s in zip(sents[0::2], sents[1::2]):
            res.append(t+s)
        return res

    def padding_data(self, texts, labels):
        pad_idx = self.word2idx["<PAD>"]
        unk_idx = self.word2idx["<UNK>"]
        # 每篇文本的句子数；每句话的长度
        sents_len_seq, words_len_seq = [], []
        text_ids, label_ids = [], []
        for text, label in zip(texts, labels):
            label_ids.append(self.label2idx[label])
            split_sents = [sent for sent in self.split_sents(text) if len(sent)>1][:self.max_sent_len]
            sents_len_seq.append(len(split_sents))  # 存放一篇文章包含句子数
            tmp_text_ids = []   # 存放一篇文章的word id
            for tmp_sent in split_sents:
                tmp_ids = [self.word2idx.get(w, unk_idx) for w in tmp_sent][:self.max_words_len]
                words_len_seq.append(len(tmp_ids))   # 每句话实际长度
                tmp_sample = [pad_idx] * self.max_words_len
                tmp_sample[:len(tmp_ids)] = tmp_ids
                tmp_text_ids.append(tmp_sample)
            if len(split_sents) < self.max_sent_len:
                tmp_text_ids.extend([[unk_idx]*self.max_words_len for _ in range(self.max_sent_len-len(split_sents))])
                words_len_seq.append(1)
            text_ids.extend(tmp_text_ids)

        return torch.tensor(text_ids), torch.tensor(sents_len_seq), torch.tensor(words_len_seq), torch.tensor(label_ids)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, item):
        return self.text_ids[item: item+self.max_sent_len], self.sent_len_seq[item], \
               self.token_len_seq[item: item+self.max_sent_len], self.label_ids[item]

