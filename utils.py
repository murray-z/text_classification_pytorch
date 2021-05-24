# -*- coding: utf-8 -*-

import json
import logging


def dump_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def get_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    # 输出到文件
    FileHandler = logging.FileHandler(log_path, mode='w')
    FileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    FileHandler.setFormatter(formatter)
    logger.addHandler(FileHandler)

    # 输出到屏幕
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
