# -*- coding: utf-8 -*-


from model import *
from data_helper import generate_batch_data
from config import *


train_path = "./data/cnews/cnews.train_digit.json"
test_path = "./data/cnews/cnews.test_digit.json"
dev_path = "./data/cnews/cnews.val_digit.json"


def train(model="textcnn"):
    model_path = "./model/model_best_{}.pth".format(model)
    if model == "textcnn":
        config = textcnn_config
        device = config["device"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        model = TextCNN(vocab_size=config["vocab_size"],
                        embed_size=config["embed_size"],
                        filter_sizes=config["filter_sizes"],
                        filter_nums=config["filter_nums"],
                        class_num=config["class_num"],
                        dropout=config["dropout"])
        model.to(device)
    elif model == "textrnn":
        config = textrnn_config
        device = config["device"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        model = TextRNN(vocab_size=config["vocab_size"],
                        embed_size=config["embed_size"],
                        num_layers=config["num_layers"],
                        hidden_size=config["hidden_size"],
                        bidirectional=config["bidirectional"],
                        class_num=config["class_num"],
                        dropout=config["dropout"])
        model.to(device)
    elif model == "textrcnn":
        config = textrcnn_config
        device = config["device"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        model = TextRCNN(vocab_size=config["vocab_size"],
                        embed_size=config["embed_size"],
                        num_layers=config["num_layers"],
                        hidden_size=config["hidden_size"],
                        bidirectional=config["bidirectional"],
                        class_num=config["class_num"],
                        dropout=config["dropout"],
                        seq_len=config["seq_len"])
        model.to(device)
    elif model == "textrnnatt":
        config = textrnnatt_config
        device = config["device"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        model = TextRnnAtt(vocab_size=config["vocab_size"],
                        embed_size=config["embed_size"],
                        num_layers=config["num_layers"],
                        hidden_size=config["hidden_size"],
                        bidirectional=config["bidirectional"],
                        class_num=config["class_num"],
                        dropout=config["dropout"])
        model.to(device)

    elif model == "textdpcnn":
        config = textdpcnn_config
        device = config["device"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        model = TextDPCNN(vocab_size=config["vocab_size"],
                        embed_size=config["embed_size"],
                        channel_size=config["channel_size"],
                        class_num=config["class_num"]
                          )
        model.to(device)


    print(model)
    best_acc = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.NLLLoss()


    for epoch in range(1, epochs+1):
        for i, (datas, labels) in enumerate(generate_batch_data(train_path, batch_size)):
            optimizer.zero_grad()
            datas = torch.from_numpy(datas).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            preds = model(datas)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                preds = torch.argmax(preds, dim=1)
                acc = torch.sum(preds == labels)/preds.size()[0]
                print("Epoch: {} Step: {} Loss: {} ACC: {}".format(epoch, i, loss.item(), acc))

        dev_acc = dev(model, data_path=dev_path, batch_size=batch_size, device=device)
        print("\nEpoch: {}  DEV_ACC: {}\n".format(epoch, dev_acc))
        if dev_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = dev_acc

    # 加载最优模型进行测试
    TEST_ACC = test(model, data_path=test_path, model_path=model_path, batch_size=batch_size, device=device)
    print("\nTEST ACC: {}".format(TEST_ACC))


def dev(model, data_path, batch_size, device):
    model.eval()
    acc_num = 0.
    all_num = 0.
    with torch.no_grad():
        for i, (datas, labels) in enumerate(generate_batch_data(data_path, batch_size)):
            datas = torch.from_numpy(datas).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            preds = model(datas)
            preds = torch.argmax(preds, dim=1)
            acc_num += torch.sum(preds == labels)
            all_num += preds.size()[0]
    model.train()
    return acc_num/all_num


def test(model, data_path, model_path, batch_size, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    acc_num = 0.
    all_num = 0.
    with torch.no_grad():
        for i, (datas, labels) in enumerate(generate_batch_data(data_path, batch_size)):
            datas = torch.from_numpy(datas).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            preds = model(datas)
            preds = torch.argmax(preds, dim=1)
            acc_num += torch.sum(preds == labels)
            all_num += preds.size()[0]
    return acc_num / all_num


if __name__ == '__main__':
    train(model="textdpcnn")