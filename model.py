# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        seq_len_sort = input[idx_sort, :, :]

        # encoder
        rnn_input = pack_padded_sequence(input_sort, seq_len_sort, batch_first=True)
        # B*T*input_size -> B*T*(num_direction*hidden_size)=B*T*K
        rnn_output, hid = self.rnn(rnn_input)
        seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True)
        rnn_output = seq_unpacked[idx_unsort, :, :]

        # attention
        # B*T*K -> B*T*attention_dim
        u = torch.tanh(self.project(rnn_output))

        # 计算attention_weight
        # B*T*att_dim B_@ B*att_dim*1 -> B*T*1  -> B*T
        # att_weight = torch.cat([(u[i, :, :].mm(self.context)).unsequeeze(0) for i in range(u.size(0))], 0)
        att_weight = torch.bmm(u, self.context.expand(u.size(0), -1, -1)).squeeze()
        # softmax: B*T -> B*1*T
        att_weight = F.softmax(att_weight, dim=1).unsqueeze(1)
        # B*1*T b_@ B*T*att_dim -> B*1*att_dim -> B*att_dim
        out_put = torch.bmm(att_weight, rnn_output).squeeze(1)
        return out_put


class HAN(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=200,
                 word_hidden_size=50, sent_hidden_size=50,
                 word_num_layers=1, sent_num_layers=1,
                 word_bidirectional=True, sent_bidirectional=True,
                 word_attention_dim=None, sent_attention_dim=None):
        """
        参考： https://github.com/qiuhuan/document-classification-pytorch
        :param num_classes: 分类标签数
        :param vocab_size: 词典大小
        :param embed_dim: 词向量维度
        :param word_hidden_size: word encoder hidden_size
        :param sent_hidden_size: sent encoder hidden_size
        :param word_num_layers: GRU 层数
        :param sent_num_layers: GRU 层数
        :param word_bidirectional:
        :param sent_bidirectional:
        :param word_attention_dim: attention 维度
        :param sent_attention_dim:
        """
        super(HAN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.word_encode = EncoderWithAttention(input_size=embed_dim,
                                                hidden_size=word_hidden_size,
                                                num_layers=word_num_layers,
                                                bidirectional=word_bidirectional,
                                                attention_dim=word_attention_dim)

        # word attention 结果作为 sentence encoder 的输入
        self.sentences_encode = EncoderWithAttention(input_size=word_hidden_size*2,
                                                     hidden_size=sent_hidden_size,
                                                     num_layers=sent_num_layers,
                                                     bidirectional=sent_bidirectional,
                                                     attention_dim=sent_attention_dim)

        self.fc = nn.Linear(sent_hidden_size*(2 if sent_bidirectional else 1), num_classes)

        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.)
            elif type(m) == nn.Embedding:
                m.weight.data.uniform_(-0.1, 0.1)

        # 初始化参数
        self.apply(init_weight())

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
                # 判断是否在gpu
                if feature_tensor.is_cuda:
                    v = torch.cat([v, patch.cuda()], dim=0)
                else:
                    v = torch.cat([v, patch], dim=0)
            # v: 1*max_sen_len*dim
            vv.append(v.unsqueeze(0))
            row_idx += length
        # B * max_sen_len * dim
        return torch.cat(vv, dim=0)

    def forward(self, input, sentence_len_seq, tokens_len_seq):
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
        logit = self.fc(sentence_encode)
        return logit

    
    
    
if __name__ == '__main__':
    from data_helper import generate_batch_data
    textcnn = TextDPCNN(3000, 128, 250, 10)
    for datas, labels in generate_batch_data("./data/cnews/cnews.val_digit.json", batch_size=10):
        textcnn(torch.from_numpy(datas))
        break









