# -*- coding: utf-8 -*-

from collections import Counter
import json
import numpy as np


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def generate_vocab_dic(train_data_path, vocab_size=3000):
    UNK = "<UNK>"
    PAD = "<PAD>"
    labels = set()
    words = []
    with open(train_data_path) as f:
        for line in f:
            lis = line.strip().split('\t')
            labels.add(lis[0])
            words.extend(list(lis[1]))
    word_count = Counter(words).most_common(vocab_size-2)
    word_list = [PAD, UNK] + [w for w, n in word_count]
    label_list = list(labels)
    l2i = {l: idx for idx, l in enumerate(label_list)}
    i2l = {idx: l for idx, l in enumerate(label_list)}

    w2idx = {w: idx for idx, w in enumerate(word_list)}
    idx2w = {idx: w for idx, w in enumerate(word_list)}

    save_json(l2i, "./data/cnews/label2idx.json")
    save_json(i2l, "./data/cnews/idx2label.json")
    save_json(w2idx, "./data/cnews/word2idx.json")
    save_json(idx2w, "./data/cnews/idx2word.json")


def get_digit_data(data_path, label2idx_path='./data/cnews/label2idx.json',
                   word2idx_path="./data/cnews/word2idx.json", max_len=200):
    label2idx = load_json(label2idx_path)
    word2idx = load_json(word2idx_path)

    labels = []
    sents = []
    with open(data_path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            labels.append(label2idx[l])
            words = list(s)[:max_len]
            sents.append([word2idx.get(w, word2idx["<UNK>"]) for w in words] + [word2idx["<PAD>"]] * (max_len-len(words)))

    save_json({"text": sents, "label": labels}, data_path.rsplit(".", 1)[0]+"_digit.json")


def generate_batch_data(data_path, batch_size=3, shuffle=True):
    datas = load_json(data_path)
    texts, labels = datas["text"], datas["label"]
    texts = np.array(texts)
    labels = np.array(labels)

    if shuffle:
        permutation_idx = np.random.permutation(len(texts))
        texts = texts[permutation_idx, :]
        labels = labels[permutation_idx]

    batch_num = int(len(texts)/batch_size)

    for i in range(batch_num):
        batch_x = texts[i*batch_size: (i+1)*batch_size, :]
        batch_y = labels[i*batch_size: (i+1)*batch_size]
        yield batch_x, batch_y



if __name__ == '__main__':
    # generate_vocab_dic("./data/cnews/cnews.train.txt")

    # for t in ["test",  "train", "val"]:
    #     get_digit_data("./data/cnews/cnews.{}.txt".format(t))

    for datas, labels in generate_batch_data("./data/cnews/cnews.train_digit.json"):
        print(datas)
        print(labels)
        break



