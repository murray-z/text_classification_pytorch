# -*- coding: utf-8 -*-

import os
import json
import argparse
from torch.utils.data import DataLoader
from model import *
from config import *
from data_helper import *
from utils import get_logger
from sklearn.metrics import classification_report


model_map = {
    "dnn": [dnn_config, CnnRnnDataSet, DNN],
    "cnn": [cnn_config, CnnRnnDataSet, CNN],
    "rnn": [rnn_config, CnnRnnDataSet, RNN],
    "rcnn": [rcnn_config, CnnRnnDataSet, RCNN],
    "rnnatt": [rnnatt_config, CnnRnnDataSet, RnnAtt],
    "dpcnn": [dpcnn_config, CnnRnnDataSet, DPCNN],
    "han": [han_config, HanDataSet, HAN],
    "bert": [bert_config, BertDataSet, Bert]
}


def dev(model, data_loader, device):
    model.eval()
    acc_num = 0.
    all_num = 0.
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_data = batch_data[:-1]
            labels = batch_data[-1]
            preds = model(*input_data)
            preds = torch.argmax(preds, dim=1)
            acc_num += torch.sum(preds == labels)
            all_num += preds.size()[0]
    return acc_num/all_num


def final_test(model, data_loader, model_path, device, label2idx):
    idx2label = {idx: label for label, idx in label2idx.items()}
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_data = batch_data[:-1]
            labels = batch_data[-1]
            preds = model(*input_data)
            preds = torch.argmax(preds, dim=1)
            labels = labels.cpu().numpy().tolist()
            preds = preds.cpu().numpy().tolist()
            true_labels.extend(labels)
            pred_labels.extend(preds)
    true_labels = [idx2label[idx] for idx in true_labels]
    pred_labels = [idx2label[idx] for idx in pred_labels]

    table = classification_report(true_labels, pred_labels, digits=4)
    return table


def main(model_type="cnn"):
    if model_type not in model_map.keys():
        print("model_type Error !")
        return

    # 数据预处理，获得词映射和标签映射
    if not os.path.exists(base_config["word2idx_path"]) or not os.path.exists(base_config["label2idx_path"]):
        pre_process_data()

    label2idx = load_json(LABEL2IDX_PATH)

    # 模型、日志文件
    if not os.path.exists(base_config["model_dir"]):
        os.mkdir(base_config["model_dir"])
    if not os.path.exists(base_config["log_dir"]):
        os.mkdir(base_config["log_dir"])
    model_path = os.path.join(base_config["model_dir"], "{}.pth".format(model_type))
    log_path = os.path.join(base_config["log_dir"], "{}.log".format(model_type))
    logger = get_logger(log_path, model_type)
    logger.info("model will save in {} ".format(model_path))
    logger.info("log will save in {} ".format(log_path))


    # 合并config
    config = base_config
    config.update(model_map[model_type][0])

    # 打印超参数
    logger.info("hyper parameters:")
    logger.info(json.dumps(config, ensure_ascii=False, indent=4))

    device = config["device"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    # 加载数据集
    dataset = model_map[model_type][1]

    train_data_loader = DataLoader(dataset(TRAIN_DATA_PATH), batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    test_data_loader = DataLoader(dataset(TEST_DATA_PATH), batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    val_data_loader = DataLoader(dataset(VAL_DATA_PATH), batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    # 加载模型
    model = model_map[model_type][2](config)
    model.to(device)

    logger.info("model parameters: ")
    logger.info(model)
    best_acc = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for i, batch_data in enumerate(train_data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_data = batch_data[:-1]
            labels = batch_data[-1]
            optimizer.zero_grad()
            preds = model(*input_data)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                preds = torch.argmax(preds, dim=1)
                acc = torch.sum(preds == labels)*1. / preds.size()[0]
                logger.info("TRAIN: Epoch: {} Step: {} Loss: {} ACC: {}".format(epoch, i+1, loss.item(), acc))

        dev_acc = dev(model, val_data_loader, device)
        logger.info("\nDEV: Epoch: {}  DEV_ACC: {}\n".format(epoch, dev_acc))
        if dev_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = dev_acc

    # 加载最优模型进行测试
    test_table = final_test(model, test_data_loader, model_path=model_path, device=device, label2idx=label2idx)
    logger.info("\nTEST TABLE: \n{}".format(test_table))




if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("model_type", default="cnn", type=str)
    args = argparse.parse_args()
    model_type = args.model_type
    main(model_type)
