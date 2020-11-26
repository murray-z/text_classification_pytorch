# -*- coding: utf-8 -*-

import torch

textcnn_config = {
    "vocab_size": 3000,
    "embed_size": 128,
    "filter_sizes": [3, 4, 5],
    "filter_nums": 100,
    "class_num": 10,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}



textrnn_config = {
    "vocab_size": 3000,
    "embed_size": 128,
    "hidden_size": 100,
    "num_layers": 2,
    "bidirectional": True,
    "class_num": 10,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}



textrcnn_config = {
    "vocab_size": 3000,
    "embed_size": 128,
    "hidden_size": 100,
    "num_layers": 2,
    "bidirectional": True,
    "class_num": 10,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.01,
    "seq_len": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


textrnnatt_config = {
    "vocab_size": 3000,
    "embed_size": 128,
    "hidden_size": 100,
    "num_layers": 2,
    "bidirectional": True,
    "class_num": 10,
    "dropout": 0.5,
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


textdpcnn_config = {
    "vocab_size": 3000,
    "embed_size": 128,
    "channel_size": 250,
    "epochs": 10,
    "batch_size": 128,
    "class_num": 10,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
