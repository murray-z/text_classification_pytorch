# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.fc1 = nn.Linear(config["embed_size"], config["hidden_size_1"])
        self.drop1 = nn.Dropout(config["dropout1"])
        self.fc2 = nn.Linear(config["hidden_size_1"], config["hidden_size_2"])
        self.drop2 = nn.Dropout(config["dropout2"])
        self.fc = nn.Linear(config["hidden_size_2"], config["class_num"])

    def forward(self, x):
        x_embed = self.embedding(x)
        x = torch.mean(x_embed, dim=1).squeeze()
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        logits = self.fc(x)
        return logits


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config["filter_nums"], (k, config["embed_size"]))
             for k in config["filter_sizes"]])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["filter_nums"] * len(config["filter_sizes"]), config["class_num"])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.embedding_layer = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.lstm = nn.LSTM(input_size=config["embed_size"],
                            hidden_size=config["hidden_size"],
                            num_layers=config["num_layers"],
                            bidirectional=config["bidirectional"],
                            batch_first=True
                            )
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["hidden_size"]*(2 if config["bidirectional"] else 1), config["class_num"])

    def forward(self, x):
        x = self.embedding_layer(x)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        logits = self.fc(x)
        return logits


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        self.embedding_layer = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.lstm = nn.LSTM(input_size=config["embed_size"],
                            hidden_size=config["hidden_size"],
                            num_layers=config["num_layers"],
                            bidirectional=config["bidirectional"],
                            batch_first=True
                            )
        self.pool = nn.MaxPool1d(config["max_doc_len"])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["hidden_size"] * (2 if config["bidirectional"] else 1) + config["embed_size"],
                            config["class_num"])


    def forward(self, x):

        embed_x = self.embedding_layer(x)
        out, _ = self.lstm(embed_x)
        cat_x = torch.cat((out, embed_x), dim=2)
        out = torch.tanh(cat_x)

        # 在时间步维度做max pooling
        out = out.permute(0, 2, 1)
        out = self.pool(out)
        out = out.squeeze()
        logits = self.fc(out)
        return logits



class RnnAtt(nn.Module):
    def __init__(self, config):
        super(RnnAtt, self).__init__()
        self.embedding_layer = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.lstm = nn.LSTM(input_size=config["embed_size"],
                            hidden_size=config["hidden_size"],
                            num_layers=config["num_layers"],
                            bidirectional=config["bidirectional"],
                            batch_first=True
                            )
        hidden_dim = config["hidden_size"]*(2 if config["bidirectional"] else 1)
        self.W = nn.Parameter(torch.randn(hidden_dim))
        self.fc = nn.Linear(hidden_dim, config["class_num"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        # batch_size * seq_len * embed_size
        embed_x = self.embedding_layer(x)
        out, _ = self.lstm(embed_x)
        x = torch.tanh(out)
        # att
        alpha = F.softmax(torch.matmul(x, self.W), dim=1).unsqueeze(-1) # batch_size*seq_len*1
        out = x * alpha # batch_size * seq_len * (hidden_size*2)
        out = torch.sum(out, dim=1)  # batch_size * (hidden_size*2)
        out = F.relu(out)
        logits = self.fc(out)  # batch_size * class_num
        return logits


class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.embedding_layer = nn.Embedding(config["vocab_size"], config["embed_size"])
        self.conv_region_embedding = nn.Conv2d(1, config["channel_size"], (3, config["embed_size"]))
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1)) # 对卷积后图像padding （左右上下）
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1)) # 对pooling后图像padding
        self.conv3 = nn.Conv2d(config["channel_size"], config["channel_size"], (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.fc = nn.Linear(config["channel_size"], config["class_num"])


    def forward(self, x):
        embed_x = self.embedding_layer(x) # batch_size * seq_len * embed_size
        embed_x = embed_x.unsqueeze(1)  # batch_size * 1 * seq_len * embedding
        # region embedding
        px = self.conv_region_embedding(embed_x)  # batch_size * channel_size * seq_len-2 * 1
        x = self.padding_conv(px)   # batch_size * channel_size * seq_len * 1
        x = F.relu(x)
        x = self.conv3(x)   # batch_size * channel_size * (seq_len-2) * 1
        x = self.padding_conv(x)  # batch_size * channel_size * seq_len * 1
        x = F.relu(x)
        x = self.conv3(x)  # batch_size * channel_size * (seq_len-2) * 1
        # region 和 conv3 残差
        x = x + px          # batch_size * channel_size * (seq_len-2) * 1
        while x.size()[-2] > 2:
            x = self._block(x)
        x = x.squeeze()
        logists = self.fc(x)
        return logists

    def _block(self, x):
        # pooling: seq//2
        x = self.padding_pool(x)
        px = self.pooling(x)
        
        # conv1
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)
        # conv2
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        # short cut
        x = x+px
        return x

    
class EncoderWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True,
                 attention_dim=None):
        """
        attention模块
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param bidirectional:
        :param attention_dim:
        """
        super(EncoderWithAttention, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=self.bidirectional,
                          num_layers=num_layers,
                          batch_first=True)
        # 方向数
        num_directions = 2 if self.bidirectional else 1
        # attention 参数
        if not attention_dim:
            attention_dim = num_directions * hidden_size

        # 转换GRU输出维度
        self.project = nn.Linear(num_directions*hidden_size, attention_dim, bias=True)
        # 用于和隐向量求 attention weight
        self.context = nn.Parameter(torch.Tensor(attention_dim, 1))
        # 初始化
        self.context.data.uniform_(-1, 1)

    def forward(self, input, seq_len):
        """
        :param input:
        :param seq_len:
        :return:
        """
        # 将输入按照长度降序排列, idx_sort:排序后输入在原始位置的索引
        _, idx_sort = torch.sort(seq_len, descending=True)
        # idx_unsort: 将排序后的输出还原
        _, idx_unsort = torch.sort(idx_sort, descending=False)

        # 将数据按照长度降序排列
        input_sort = input[idx_sort, :, :]
        seq_len_sort = seq_len[idx_sort].cpu()

        # encoder
        rnn_input = pack_padded_sequence(input_sort, seq_len_sort, batch_first=True)
        # B*T*input_size -> B*T*(num_direction*hidden_size)=B*T*K
        rnn_output, hid = self.rnn(rnn_input)
        seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True)
        rnn_output = seq_unpacked[idx_unsort, :, :]

        # rnn_output, hid = self.rnn(input)

        # attention
        # B*T*K -> B*T*attention_dim
        u = torch.tanh(self.project(rnn_output))

        # 计算attention_weight
        # B*T*att_dim B_@ B*att_dim*1 -> B*T*1  -> B*T
        att_weight = torch.cat([(u[i, :, :].mm(self.context)).unsqueeze(0) for i in range(u.size(0))], 0).squeeze(2)
        # att_weight = torch.bmm(u, self.context.expand(u.size(0), -1, -1)).squeeze()
        # softmax: B*T -> B*1*T
        att_weight = F.softmax(att_weight, dim=1).unsqueeze(1)
        # B*1*T b_@ B*T*att_dim -> B*1*att_dim -> B*att_dim
        out_put = torch.bmm(att_weight, rnn_output).squeeze(1)
        return out_put


class HAN(nn.Module):
    def __init__(self, config):
        """
        参考： https://github.com/qiuhuan/document-classification-pytorch
        """
        super(HAN, self).__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["embed_size"])

        self.word_encode = EncoderWithAttention(input_size=config["embed_size"],
                                                hidden_size=config["word_hidden_size"],
                                                num_layers=config["word_num_layers"],
                                                bidirectional=config["word_bidirectional"],
                                                attention_dim=config["word_attention_dim"])

        # word attention 结果作为 sentence encoder 的输入
        self.sentences_encode = EncoderWithAttention(input_size=config["word_hidden_size"]*2,
                                                     hidden_size=config["sent_hidden_size"],
                                                     num_layers=config["sent_num_layers"],
                                                     bidirectional=config["sent_bidirectional"],
                                                     attention_dim=config["sent_attention_dim"])

        self.fc = nn.Linear(config["sent_hidden_size"]*(2 if config["sent_bidirectional"] else 1), config["class_num"])

        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.)
            elif type(m) == nn.Embedding:
                m.weight.data.uniform_(-0.1, 0.1)

        # 初始化参数
        self.apply(init_weight)

    def resize2batchSentences(self, feature_tensor, sen_len_seq):
        """
        将经过word att 后的特征转为 sent att的输入
        :param feature_tensor: [B_all_sent, dim]: 一个batch包含的所有文本的句子数 * hidden_size
        :param sen_len_seq: 每篇文章包含句子数
        :return:
        """
        # dim: sent att的输入；max_sen_len: batch中文本包含的最多句子
        dim, max_sen_len = feature_tensor.size(1), torch.max(sen_len_seq)

        vv, row_idx = [], 0
        for length in sen_len_seq:
            length = int(length)
            # 这里是一篇文章的所有句子向量
            v = feature_tensor[row_idx: row_idx + length, :]
            # 将文本包含句子数目补齐
            if length < max_sen_len:
                patch = torch.zeros(max_sen_len - length, dim)
                v = torch.cat([v, patch.to(v.device)], dim=0)
            # v: 1*max_sen_len*dim
            vv.append(v.unsqueeze(0))
            row_idx += length
        # B * max_sen_len * dim
        return torch.cat(vv, dim=0)

    def forward(self, input, sentence_len_seq, tokens_len_seq):
        # 修正input和tokens_len_seq形状
        input = input.view((-1, input.size(2)))
        tokens_len_seq = tokens_len_seq.flatten()

        # 按照sentence_len_seq，保留真实的文本
        batch_size = int(input.size(0)/len(sentence_len_seq))
        new_input = torch.zeros((torch.sum(sentence_len_seq).item(), input.size(1)))
        new_tokens_len = torch.zeros(torch.sum(sentence_len_seq))
        start1 = 0
        start2 = 0
        for idx, length in enumerate(sentence_len_seq.cpu().numpy().tolist()):
            new_input[start1: start1+length, :] = input[start2: start2+length, :]
            new_tokens_len[start1: start1+length] = tokens_len_seq[start2: start2+length]
            start1 += length
            start2 += batch_size

        input = new_input.int().to(input.device)
        tokens_len_seq = new_tokens_len.int().to(input.device)

        # B_Ts * Tw  -> B_Ts * Tw * embed_size
        embed = self.embed(input)
        # B_Ts*Tw*embed_size -> B_Ts * att_dim
        word_encode = self.word_encode(embed, tokens_len_seq)

        # 将句子Batch转成篇章Batch
        # B_Ts*att_dim -> B*Ts*word_att_dim
        word_encode = self.resize2batchSentences(word_encode, sentence_len_seq)

        # B*Ts*word_att_dim -> B*sent_att_dim
        sentence_encode = self.sentences_encode(word_encode, sentence_len_seq)

        # logits
        logits = self.fc(sentence_encode)
        return logits


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained(config["pre_bert_model"])
        self.fc = nn.Linear(config["hidden_size"], config["class_num"])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
    









