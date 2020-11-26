# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, filter_sizes, filter_nums, class_num, dropout):
        super(TextCNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv2d(1, filter_nums, (filter_sizes[0], embed_size))
        self.conv2 = nn.Conv2d(1, filter_nums, (filter_sizes[1], embed_size))
        self.conv3 = nn.Conv2d(1, filter_nums, (filter_sizes[2], embed_size))
        self.fc = nn.Linear(filter_nums*len(filter_sizes), class_num)
        self.dropout = nn.Dropout(dropout)

    def conv_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x.squeeze(3))
        x = nn.MaxPool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)
        x1 = self.conv_pool(x, self.conv1)
        x2 = self.conv_pool(x, self.conv2)
        x3 = self.conv_pool(x, self.conv3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        logit = F.log_softmax(x, dim=1)
        return logit


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, class_num):
        super(TextRNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True
                            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else 1, class_num)

    def forward(self, x):
        x = self.embedding_layer(x)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.fc(x)
        x = self.dropout(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class TextRCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, class_num, seq_len):
        super(TextRCNN, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True
                            )
        self.pool = nn.MaxPool1d(seq_len)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear((hidden_size * 2 if bidirectional else 1)+embed_size, class_num)


    def forward(self, x):
        embed_x = self.embedding_layer(x)
        out, _ = self.lstm(embed_x)
        cat_x = torch.cat((out, embed_x), dim=2)
        out = torch.tanh(cat_x)
        out = out.permute(0, 2, 1)
        out = self.pool(out)
        out = out.squeeze(2)
        out = self.fc(out)
        logits = F.log_softmax(out, dim=1)
        return logits



class TextRnnAtt(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, class_num):
        super(TextRnnAtt, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True
                            )
        self.tanh1 = nn.Tanh()
        self.W = nn.Parameter(torch.zeros(hidden_size*2))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden_size*2, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch_size * seq_len * embed_size
        embed_x = self.embedding_layer(x)
        # out: batch_size * seq_le * (hidden_size*2)
        H, _ = self.lstm(embed_x)
        M = self.tanh1(H)
        # att
        alpha = F.softmax(torch.matmul(M, self.W), dim=1).unsqueeze(-1) # batch_size*seq_len*1
        out = H * alpha # batch_size * seq_len * (hidden_size*2)
        out = torch.sum(out, dim=1)  # batch_size * (hidden_size*2)
        out = F.relu(out)
        out = self.fc(out)  # batch_size * class_num
        logits = F.log_softmax(out, dim=1)
        return logits


class TextDPCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, channel_size, class_num):
        super(TextDPCNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.conv_region_embedding = nn.Conv2d(1, channel_size, (3, embed_size))
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1)) # 对卷积后图像padding （左右上下）
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1)) # 对pooling后图像padding
        self.conv3 = nn.Conv2d(channel_size, channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.fc = nn.Linear(channel_size, class_num)


    def forward(self, x):
        embed_x = self.embedding_layer(x) # batch_size * seq_len * embed_size
        embed_x = embed_x.unsqueeze(1)  # batch_size * 1 * seq_len * embedding
        # region embedding
        x = self.conv_region_embedding(embed_x)  # batch_size * channel_size * seq_len-2 * 1
        x = self.padding_conv(x)   # batch_size * channel_size * seq_len * 1
        x = F.relu(x)
        x = self.conv3(x)   # batch_size * channel_size * (seq_len-2) * 1
        x = self.padding_conv(x)  # batch_size * channel_size * seq_len * 1
        x = F.relu(x)
        x = self.conv3(x)  # batch_size * channel_size * (seq_len-2) * 1
        while x.size()[-2] > 2:
            x = self._block(x)
        x = x.squeeze()
        logists = F.log_softmax(self.fc(x), dim=1)
        return logists

    def _block(self, x):
        # pooling
        x = self.padding_pool(x)
        px = self.pooling(x)
        # conv
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)
        # short cut
        x = x+px
        return x

if __name__ == '__main__':
    from data_helper import generate_batch_data
    textcnn = TextDPCNN(3000, 128, 250, 10)
    for datas, labels in generate_batch_data("./data/cnews/cnews.val_digit.json", batch_size=10):
        textcnn(torch.from_numpy(datas))
        break









