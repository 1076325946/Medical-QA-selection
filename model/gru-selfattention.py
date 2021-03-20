#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
# import argparse #python标准库里面用来处理命令行参数的库
import os  #导入操作系统接口模块
import random  #python中导入随机函数random包
import time  #导入时间模块

import numpy as np 
import torchtext #torchtext这一文本处理神器，可以方便的对文本进行预处理，例如截断补长、构建词表等
import torch
#from tensorboardX import SummaryWriter#可以将tensorboard应用到Pytorch中，用于Pytorch的可视化
from tqdm import tqdm#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)

# from args import parse_args

#model_bao_peizhi
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from sys import platform
#utility_peizhi
import pandas as pd
import math


# In[ ]:


class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias, 
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # for linux
        if platform == "linux" or platform == "linux2":
            reordered_outputs = outputs.index_select(0, restoration_idx)
        # for win10
        else:
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch
    
def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask	

def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return mask * tensor + values_to_add

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

    
def get_accuracy(qids, predictions, labels, topk=1):
    tmp = list(zip(qids, predictions, labels))
    random.shuffle(tmp)
    qids, predictions, labels = zip(*tmp)
    qres = {}
    for i,qid in enumerate(qids):
        pre = predictions[i]
        label = labels[i]
        if qid in qres:
            qres[qid]['labels'].append(label)
            qres[qid]['predictions'].append(pre)
        else:
            qres[qid] = {'labels': [label], 'predictions': [pre]}
    correct = 0
    for qid,res in qres.items():
        label_index = [i for i,v in enumerate(res['labels']) if v == 1]
        pre_index = sorted(enumerate(res['predictions']), key=lambda x:x[1], reverse=True)[:topk]
        is_correct = [(k,v) for k,v in pre_index if k in label_index]
        if len(is_correct) > 0:
            correct += 1
    return correct / len(qres)

def get_dataset():
    fix_length = 400
    question = pd.read_csv('./questions.csv', index_col='que_id')
    answer = pd.read_csv('./answers.csv', index_col='ans_id')
    train_candidates = pd.read_csv('./train_candidates.txt')
    dev_candidates = pd.read_csv('./dev_candidates.txt', header=1,
                                 names=['question_id', 'ans_id', 'num', 'label'])
    test_candidates = pd.read_csv('./test_candidates.txt', header=1,
                                  names=['question_id', 'ans_id', 'num', 'label'])
    
    # 记录问题和答案的id到文本的映射
    qid2text = {index: item['content'] for index, item in question.iterrows()}
    aid2text = {index: item['content'] for index, item in answer.iterrows()}
    
    # 定义数据loader
    ID_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = torchtext.data.Field(batch_first=True, tokenize=lambda x: list(x), fix_length=fix_length,
                                      include_lengths=True)
    LABEL_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    
    # 问题
    examples = []
    fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
    for que_id, content in question.content.items():
        example_list = [que_id, content]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    question_dataset = torchtext.data.Dataset(examples, fields)

    # 答案
    examples = []
    fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
    for ans_id, content in answer.content.items():
        example_list = [ans_id, content]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    answer_dataset = torchtext.data.Dataset(examples, fields)
    
    # 训练集
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('pos_answer', TEXT_FIELD), ('neg_answer', TEXT_FIELD)]
    for question_id, pos_ans_id, neg_ans_id in zip(train_candidates.question_id.values,
                                                   train_candidates.pos_ans_id.values,
                                                   train_candidates.neg_ans_id.values):
        example_list = [question_id, qid2text[question_id], aid2text[pos_ans_id], aid2text[neg_ans_id]]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    train_dataset = torchtext.data.Dataset(examples, fields)

    # 验证集
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
    for question_id, ans_id, label in zip(dev_candidates.question_id.values, dev_candidates.ans_id.values,
                                          dev_candidates.label.values):
        example_list = [question_id, qid2text[question_id], aid2text[ans_id], label]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    dev_dataset = torchtext.data.Dataset(examples, fields)
    
    # 测试集
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
    for question_id, ans_id, label in zip(test_candidates.question_id.values, test_candidates.ans_id.values,
                                          test_candidates.label.values):
        example_list = [question_id, qid2text[question_id], aid2text[ans_id], label]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    test_dataset = torchtext.data.Dataset(examples, fields)
    
    # 载入预训练的词向量
    cache = 'mycache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vector_file = './sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    pre_vectors = torchtext.vocab.Vectors(name=vector_file,cache=cache)
        
    # 构建词表 时间较长 3分钟左右
    TEXT_FIELD.build_vocab(question_dataset, answer_dataset, vectors=pre_vectors)

    vocab = TEXT_FIELD.vocab  # 词表
    vectors = TEXT_FIELD.vocab.vectors  # 预训练的词向量
    return train_dataset, dev_dataset, test_dataset, vocab, vectors

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """
    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses       


# In[ ]:


# 获取数据集
train_dataset, dev_dataset, test_dataset, vocab, vectors = get_dataset()
#词向量维度设置为300维
vectors_dim = 300 if vectors is None else vectors.size(1)


# In[ ]:


# 创建迭代器
#data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
#####args = parse_args()
# 初始化随机数种子，以便于复现实验结果
start_epoch = 1
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
lr = 0.001
skip_training=False
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(1234)
device = torch.device("cuda" if USE_CUDA else "cpu")
model_dir = "../sunyu/y/model"
train_loader = torchtext.data.BucketIterator(train_dataset, 64, device=device, train=True,
                                             shuffle=True, sort=False, repeat=False)
dev_loader = torchtext.data.BucketIterator(dev_dataset, 64, device=device, train=False,
                                           shuffle=False, sort=False, repeat=False)
test_loader = torchtext.data.BucketIterator(test_dataset, 64, device=device, train=False,
                                            shuffle=False, sort=False, repeat=False)


# In[ ]:


def pack_for_rnn_seq(embed_seq, seq_len):
    """
    the format of seq is batch first.
    :param embed_seq:
    :param seq_len:
    :return:
    """
    _, idx_sort = torch.sort(seq_len, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)

    embed_seq = embed_seq.index_select(0, idx_sort)
    sent_len = seq_len[idx_sort]
    packed_seq = nn.utils.rnn.pack_padded_sequence(embed_seq, sent_len, batch_first=True)
    return packed_seq, idx_unsort


def unpack_from_rnn_seq(packed_seq, idx_unsort, total_length=None):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True, total_length=total_length)
    unsort_seq = unpacked_seq.index_select(0, idx_unsort)
    return unsort_seq


def fit_seq_max_len(seq, seq_len):
    '''
    自动匹配序列的最大长度
    由于使用torchtext中的fix_length函数，得到的序列长度都是400维，而大部分情况下，
    一个batch的最大长度往往不够400维，使用此函数能将序列的最大长度固定到一个batch中的最大长度。
    :param embed_seq:
    :param seq_len:
    :return:
    '''
    packed_seq, idx_unsort = pack_for_rnn_seq(seq, seq_len)
    seq = unpack_from_rnn_seq(packed_seq, idx_unsort)
    return seq


def auto_rnn_bilstm(lstm, embed_seq, lengths, fix_length=False):
    packed_seq, idx_unsort = pack_for_rnn_seq(embed_seq, lengths)
    output, (hn, cn) = lstm(packed_seq)
    total_length = None
    if fix_length:
        total_length = embed_seq.size(1)
    unpacked_output = unpack_from_rnn_seq(output, idx_unsort, total_length)
    hn_unsort = hn.index_select(1, idx_unsort)
    cn_unsort = cn.index_select(1, idx_unsort)
    return unpacked_output, (hn_unsort, cn_unsort)


def auto_rnn_bigru(gru, embed_seq, lengths, fix_length=False):
    """
    自动对变长序列做pack和pad操作
    :param gru:
    :param embed_seq:
    :param lengths:
    :param fix_length: 是否固定输出的结果，默认会变长序列中最长那个序列长度
    :return:
    """
    packed_seq, idx_unsort = pack_for_rnn_seq(embed_seq, lengths)
    output, hn = gru(packed_seq)
    total_length = None
    if fix_length:
        total_length = embed_seq.size(1)
    unpacked_output = unpack_from_rnn_seq(output, idx_unsort, total_length)
    hn_unsort = hn.index_select(1, idx_unsort)
    return unpacked_output, hn_unsort

class BiGRUCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=200, cnn_channel=500, dropout_r=0, embed_weight=None):
        super().__init__()
        print({
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'cnn_channel': cnn_channel,
            'dropout_r': dropout_r
        })
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.gru_qa = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                             num_layers=1, batch_first=True, bidirectional=True)
        
        self.ST = SelfAttention(hidden_size=400, num_attention_heads=8, dropout_prob=0.2)
        
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        # 合并问题和答案，加速网络运行
        bs = question.size(0)
        sent = torch.cat([question, answer], dim=0)
        sent_len = torch.cat([question_length, answer_length], dim=0)
        sent = fit_seq_max_len(sent, sent_len)
        q_mask = get_mask(sent, sent_len).to(device)
        
        embed = self.embed(sent)  # (bs, sent, vector)
        gru_sent, hn_sent = auto_rnn_bigru(self.gru_qa, embed, sent_len)# [b,s,d]
        
        context = self.ST(gru_sent,q_mask)
        context_pool = torch.max(context, dim=1)[0]
        sim = self.similarity(context_pool[:bs], context_pool[bs:])
        return sim
# class ESIM(nn.Module):
#     def __init__(self, hidden_size=200, embeddings=None, dropout=0.5, num_classes=2, device="gpu"):
#         super(ESIM, self).__init__()
#         self.embedding_dim = embeddings.shape[1]
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.dropout = dropout
#         self.device = device
#         self.word_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])

#         if self.dropout:
#             self.rnn_dropout = RNNDropout(p=self.dropout)
#         self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
# #         self.projection = nn.Sequential(nn.Linear(7*2*self.hidden_size, self.hidden_size), 
# #                                         nn.ReLU())
# #         self.projection2 = nn.Sequential(nn.Linear(4*2*self.hidden_size, 2*self.hidden_size), 
# #                                         nn.ReLU())
# #         self.lin = Self_attention(hidden_size,hidden_size)
#         self.attention = SoftmaxAttention()
#         self.attention_self = SelfAttention(hidden_size=2*self.hidden_size, num_attention_heads=8, dropout_prob=self.dropout)
#         self.second_rnn = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True)
#         self.similarity = nn.CosineSimilarity(dim=1)
    
        
#     def forward(self, question, answer):
#         q1, q1_lengths = question
#         q2, q2_lengths = answer
#         q1_mask = get_mask(q1, q1_lengths).to(self.device)
#         q2_mask = get_mask(q2, q2_lengths).to(self.device)
#         q1_embed = self.word_embedding(q1)
#         q2_embed = self.word_embedding(q2)
#         if self.dropout:
#             q1_embed = self.rnn_dropout(q1_embed)
#             q2_embed = self.rnn_dropout(q2_embed)
#         # 双向lstm编码
#         q1_encoded = self.first_rnn(q1_embed, q1_lengths)
#         q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        
#         q1_selfat = self.attention_self(q1_encoded,q1_mask)
#         q2_selfat = self.attention_self(q2_encoded,q2_mask)
#         # atention
#         #q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
#         # concat
#         #q1_combined = torch.cat([q1_encoded, q1_aligned,  q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
#         #q2_combined = torch.cat([q2_encoded, q2_aligned,  q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)

# #         q1_sigmoid = self.lin(q1_combined)
# #         q1_clean = torch.mul(q1_sigmoid, q1_combined)
# #         q2_sigmoid = self.lin(q2_combined)
# #         q2_clean = torch.mul(q2_sigmoid, q2_combined)
# #         # 映射一下
# #         projected_q1 = self.projection(q1_clean)
# #         projected_q2 = self.projection(q2_clean)
# #         if self.dropout:
# #             projected_q1 = self.rnn_dropout(projected_q1)
# #             projected_q2 = self.rnn_dropout(projected_q2)
#         # 再次经过双向RNN
# #         q1_compare = self.second_rnn(projected_q1, q1_lengths)
# #         q2_compare = self.second_rnn(projected_q2, q2_lengths)
        
#         # 平均池化 + 最大池化
#         q1_avg_pool = torch.sum(q1_selfat * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
#         q2_avg_pool = torch.sum(q2_selfat * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
#         q1_max_pool, _ = replace_masked(q1_selfat, q1_mask, -1e7).max(dim=1)
#         q2_max_pool, _ = replace_masked(q2_selfat, q2_mask, -1e7).max(dim=1)
#         merged_q1 = torch.cat([q1_avg_pool,q1_max_pool], dim=1)
#         merged_q2 = torch.cat([q2_avg_pool,q2_max_pool], dim=1)
        
        
#         sim = self.similarity(merged_q1, merged_q2)
#         return sim
    
class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:   # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads    # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)    # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   # 
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0   # padding的token置为-10000，exp(-1w)=0
        
        # 线性变换
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # [bs, 8, seqlen, 16]

        
        # 计算query与title之间的点积注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer    # [bs, seqlen, 128] 得到输出
    


# In[ ]:


def train_epoch(epoch, data_loader, model, optimizer, loss_fn, device):
    """
    进行一次迭代
    """
    model.train()
    pbar = tqdm(data_loader, desc='Train Epoch {}'.format(epoch))
    total_loss = []
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        target = torch.ones(batch.batch_size, requires_grad=True).to(device)
        pos_sim = model(batch.question, batch.pos_answer)
        neg_sim = model(batch.question, batch.neg_answer)
        loss = loss_fn(pos_sim, neg_sim, target)
        total_loss.append(loss.item())
        pbar.set_postfix(batch_loss=loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(total_loss)


def evaluate(date_loader, model, topk):
    """
    在dev上进行测试
    """
    model.eval()
    pbar = tqdm(date_loader, desc=f'Evaluate')
    # 记录预测结果，计算Top-1正确率
    qids = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in pbar:
            qids.extend(batch.id.cpu().numpy())
            true_labels.extend(batch.label.cpu().numpy())
            output = model(batch.question, batch.answer)
            predictions.extend(output.cpu().numpy())
    if isinstance(topk, int):
        accuracy = get_accuracy(qids, predictions, true_labels, 1)
        return accuracy
    elif isinstance(topk, list):
        accuracies = {}
        for i in topk:
            accuracy = get_accuracy(qids, predictions, true_labels, i)
            accuracies[i] = accuracy
        return accuracies
    else:
        raise ValueError('Error topk')


def run():

    # 创建模型，优化器，损失函数
#     model = ESIM(hidden_size=200, embeddings=vectors, dropout=0.5, 
#                  num_classes=2, device=device).to(device)
    model = BiGRUCNN(vocab_size=len(vocab), embedding_dim=vectors_dim, hidden_size=200,dropout_r=0.2, embed_weight=vectors).to(device)
    # 为特定模型指定特殊的优化函数
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
#         optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
#     else:
#         raise ValueError("--optimizer is unknown")
    loss_fn = torch.nn.MarginRankingLoss(margin=0.05)
    architecture = model.__class__.__name__
#     # 载入以训练的数据
#     if args.resume_snapshot:
#         state = torch.load(args.resume_snapshot)
#         model.load_state_dict(state['model'])
#         optimizer.load_state_dict(state['optimizer'])
#         epoch = state['epoch']
#         start_epoch = state['epoch'] + 1
#         if 'best_dev_score' in state:
#             # 适配旧版本保存的模型参数
#             dev_acc = state['best_dev_score']
#             test_acc = 0
#         else:
#             dev_acc = state['dev_accuracy']
#             test_acc = state['test_accuracy']
#         logger.info(f"load state {args.resume_snapshot}, dev accuracy {dev_acc}, test accuracy {test_acc}")
#     # 记录参数
#     with open(f'{output_dir}/arguments.csv', 'a') as f:
#         for k, v in vars(args).items():
#             f.write(f'{k},{v}\n')
#     # 将日志写入到TensorBoard中
#     writer = SummaryWriter(output_dir)
#     # 记录模型的计算图
#     try:
#         q = torch.randint_like(torch.Tensor(1, args.fix_length), 2, 100, dtype=torch.long)
#         ql = torch.Tensor([args.fix_length]).type(torch.int)
#         writer.add_graph(model, ((q, ql), (q, ql)))
#     except Exception as e:
#         logger.error("Failed to save model graph: {}".format(e))
#         # exit()
    # 开始训练
    best_dev_score = -1  # 记录最优的结果
    best_test_score = -1  # 记录最优的结果
    prev_loss = 0
    # 自动调整学习率
    # TODO:暂不启用，Adam已经能够自动调整学习率了
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
    #                                                        patience=args.patience, verbose=True)
    if not skip_training:
        for epoch in range(start_epoch, start_epoch + 100):
            
            # train epoch
            loss = train_epoch(epoch, train_loader, model, optimizer, loss_fn, device)
            print(f'Train Epoch {epoch}: loss={loss}')
            # evaluate
            dev_accuracy = evaluate(dev_loader, model, 1)
            print(f'Evaluation metrices: dev accuracy = {100. * dev_accuracy}%')
            
            # 进行测试
            test_accuracy = evaluate(test_loader, model, 1)
            print(f'Evaluation metrices: test accuracy = {100. * test_accuracy}%')
           

            # 保存模型
            save_state = {'epoch': epoch, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy,
                          'architecture': architecture, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(save_state, f'{model_dir}/{architecture}_epoch_{epoch}.pth')

           

            if abs(prev_loss - loss) <= -1:
                break
            prev_loss = loss
    else:
        # 进行测试
        dev_accuracies = evaluate(dev_loader, model,1)
        for k in 1:
            print(f'Evaluation metrices: top-{k} dev accuracy = {dev_accuracies[k]}%')

        test_accuracies = evaluate(test_loader, model, 1)
        for k in 1:            
            print(f'Evaluation metrices: top-{k} test accuracy = {test_accuracies[k]}%')

            print(f'Evaluation metrices: top-{k} test accuracy = {test_accuracies[k]}%')
    # 保存embedding到tensorboard做可视化
    # writer.add_embedding(model.embed.weight.detach(), vocab.itos, global_step=epoch)


if __name__ == '__main__':
    run()

