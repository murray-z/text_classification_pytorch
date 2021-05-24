# -*- coding: utf-8 -*-

import torch


base_config = {
    "train_data_path": "./data/cnews/cnews.train.txt",
    "test_data_path": "./data/cnews/cnews.test.txt",
    "val_data_path": "./data/cnews/cnews.val.txt",
    "word2idx_path": "./data/word2idx.json",       # 词和id映射
    "label2idx_path": "./data/label2idx.json",     # 标签和id映射
    "vocab_size": 5000,    # embedding 词典大小
    "embed_size": 200,     # embedding向量大小
    "max_sent_len": 10,    # DPCNN:每篇文章有多少句子参与计算
    "max_words_len": 50,   # DPCNN:每句话最大长度
    "max_doc_len": 200,    # 整篇文章最大长度
    "num_workers": 10,     # 加载数据时并行度
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "class_num": 10,       # 分类类别数
    "pre_bert_model": "bert-base-chinese",   # transformer加载bert名称
    "model_dir": "./models",  # 模型存放路径
    "log_dir": "./logs"       # 日志存放路径
}


cnn_config = {
    "filter_sizes": [2, 3, 4],
    "filter_nums": 100,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001,
}


rnn_config = {
    "hidden_size": 100,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.01
}



rcnn_config = {
    "hidden_size": 100,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001
}


rnnatt_config = {
    "hidden_size": 100,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001
}


dpcnn_config = {
    "channel_size": 250,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001
}


bert_config = {
    "dropout": 0.1,
    "hidden_size": 768,
    "batch_size": 16,
    "epochs": 3
}


han_config = {
    "word_hidden_size": 100,
    "sent_hidden_size": 100,
    "word_num_layers": 1,
    "sent_num_layers": 1,
    "word_bidirectional": True,
    "sent_bidirectional": True,
    "word_attention_dim": None,
    "sent_attention_dim": None,
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.001
}