
# coding: utf-8

# In[ ]:


import os  #导入操作系统接口模块
import random  #python中导入随机函数random包
import time  #导入时间模块
import numpy as np 
import torchtext #torchtext这一文本处理神器，可以方便的对文本进行预处理，例如截断补长、构建词表等
import torch
from tqdm import tqdm#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from sys import platform
import pandas as pd
import math
import gc


fix_length = 400
question = pd.read_csv('./questions.csv', index_col='que_id')
answer = pd.read_csv('./answers.csv', index_col='ans_id')
dev_candidates = pd.read_csv('./dev_candidates.txt', header=1,
                                 names=['question_id', 'ans_id', 'num', 'label'])
test_candidates = pd.read_csv('./test_candidates.txt', header=1,
                                  names=['question_id', 'ans_id', 'num', 'label'])
qid2text = {index: item['content'] for index, item in question.iterrows()}
aid2text = {index: item['content'] for index, item in answer.iterrows()}
ID_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
TEXT_FIELD = torchtext.data.Field(batch_first=True, tokenize=lambda x: list(x), fix_length=fix_length,include_lengths=True)
LABEL_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)

examples = []
fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
pbar1 = tqdm(question.content.items(), desc='Loading question...')
for que_id, content in pbar1:
    example_list = [que_id, content]
    example = torchtext.data.Example.fromlist(example_list, fields)
    examples.append(example)
question_dataset = torchtext.data.Dataset(examples, fields)
# 答案
examples = []
fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
pbar2 = tqdm(answer.content.items(), desc='Loading answer...')
for ans_id, content in pbar2:
    example_list = [ans_id, content]
    example = torchtext.data.Example.fromlist(example_list, fields)
    examples.append(example)
answer_dataset = torchtext.data.Dataset(examples, fields)
    
# 验证集
examples = []
fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
pbar4 = tqdm(zip(dev_candidates.question_id.values, dev_candidates.ans_id.values,dev_candidates.label.values), desc='Loading dev...')
for question_id, ans_id, label in pbar4:
    example_list = [question_id, qid2text[question_id], aid2text[ans_id], label]
    example = torchtext.data.Example.fromlist(example_list, fields)
    examples.append(example)
dev_dataset = torchtext.data.Dataset(examples, fields)
# 测试集
examples = []
fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
pbar5 = tqdm(zip(test_candidates.question_id.values, test_candidates.ans_id.values,test_candidates.label.values), desc='Loading test...')
for question_id, ans_id, label in pbar5:
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
    return tensor * mask + values_to_add

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

def mean_max(x):
    return torch.mean(x, dim=2), torch.max(x, dim=2)[0]

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

#词向量维度设置为300维
vectors_dim = 300 if vectors is None else vectors.size(1)
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
model_dir = "/home/test/liuluyao/y/model"
dev_loader = torchtext.data.BucketIterator(dev_dataset, 64, device=device, train=False,shuffle=False, sort=False, repeat=False)
test_loader = torchtext.data.BucketIterator(test_dataset, 64, device=device, train=False,shuffle=False, sort=False, repeat=False)


# In[ ]:


class ESIM(nn.Module):
    def __init__(self, hidden_size=200, embeddings=None, dropout=0.5, num_classes=2, device="gpu"):
        super(ESIM, self).__init__()
        self.embedding_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.word_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1],_weight=embeddings)

        if self.dropout:
            self.rnn_dropout = RNNDropout(p=0.1)
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, self.hidden_size), 
                                        nn.LeakyReLU())
        self.attention = SoftmaxAttention()
        self.ST = SelfAttention(d_hid=400)
        self.linear = nn.Linear(4*4*self.hidden_size, 4*self.hidden_size, bias=True)
        self.linear1 = nn.Linear(4*2*self.hidden_size, 4*self.hidden_size, bias=False)
        self.second_rnn = Seq2SeqEncoder(nn.GRU, self.hidden_size, self.hidden_size, bidirectional=True)
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, question, answer):
        
        q1, q1_lengths = question
        q2, q2_lengths = answer
        q1_mask = get_mask(q1, q1_lengths).to(self.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.device)
        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        # 双向lstm编码
        q1_encoded = self.first_rnn(q1_embed, q1_lengths)
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        contextq1 = self.ST(q1_encoded,q1_lengths)
        contextq2 = self.ST(q2_encoded,q2_lengths)
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
        # concat
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        # 映射一下
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        # 再次经过双向RNN
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        q1_pool = torch.max(q1_compare, dim=1)[0]
        q2_pool = torch.max(q2_compare, dim=1)[0]
        q1_last = torch.cat([contextq1,q1_pool], dim=-1)
        q2_last = torch.cat([contextq2,q2_pool], dim=-1)
        # 平均池化 + 最大池化
        #q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
        #q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
        #q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        #q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)
        #merged_q1 = torch.cat([q1_avg_pool, q1_max_pool], dim=1)
        #merged_q2 = torch.cat([q2_avg_pool, q2_max_pool], dim=1)
        
        m = torch.cat([q1_last, q2_last, q1_last * q2_last,q1_last - q2_last], dim=-1)
        mamb = nn.Tanh()(self.linear(m))
        g_a = torch.cat([q1_last, q2_last], dim=-1)
        gate_a = nn.Sigmoid()(self.linear1(g_a))
        g_b = torch.cat([q2_last, q1_last], dim=-1)
        gate_b = nn.Sigmoid()(self.linear1(g_b))
        
        output_a = torch.mul(gate_a, mamb) + (1 - gate_a) * q1_last
        output_b = torch.mul(gate_b, mamb) + (1 - gate_b) * q2_last
        sim = self.similarity(output_a, output_b)
        return sim

class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, lens):
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)
        scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context # 既然命名为context就应该是整句的表示


total_loss = []
def train_epoch(epoch, data_loader,model, optimizer, loss_fn, device,i):
    model.train()
    pbar = tqdm(data_loader, desc='Train Epoch {}.{}'.format(epoch,i))
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        target = torch.ones(batch.batch_size, requires_grad=True).to(device)
        pos_sim = model(batch.question, batch.pos_answer)
        neg_sim = model(batch.question, batch.neg_answer)
        loss = loss_fn(pos_sim, neg_sim, target)
        total_loss.append(loss.item())
        #pbar.set_postfix(batch_loss=loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx%100==0:
            print(np.mean(total_loss))
    return np.mean(total_loss)

def evaluate(date_loader, model, topk):
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
    model = ESIM(hidden_size=200, embeddings=vectors, dropout=0.2, num_classes=2, device=device).to(device)
    model = nn.DataParallel(model,device_ids=[0,1]) 
    # 为特定模型指定特殊的优化函数
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn = torch.nn.MarginRankingLoss(margin=0.05)
    architecture = model.__class__.__name__
    # 开始训练
    best_dev_score = -1  # 记录最优的结果
    best_test_score = -1  # 记录最优的结果
    prev_loss = 0
    if not skip_training:
        for epoch in range(start_epoch, start_epoch + 100):
            # train epoch
            for i in range(5):
                train_candidates = pd.read_csv('./train_candidates{}.txt'.format(i))
                examples=[]
                fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('pos_answer', TEXT_FIELD), ('neg_answer', TEXT_FIELD)]
                pbar3 = tqdm(zip(train_candidates.question_id.values,train_candidates.pos_ans_id.values,train_candidates.neg_ans_id.values), desc='Loading train1.{}'.format(i))
                for question_id, pos_ans_id, neg_ans_id in pbar3:
                    example_list = [question_id, qid2text[question_id], aid2text[pos_ans_id], aid2text[neg_ans_id]]
                    example = torchtext.data.Example.fromlist(example_list, fields)
                    examples.append(example)
                train_dataset = torchtext.data.Dataset(examples, fields)
                train_loader = torchtext.data.BucketIterator(train_dataset, 64, device=device, train=True,shuffle=True, sort=False, repeat=False)
                loss = train_epoch(epoch, train_loader,model, optimizer, loss_fn, device,i)
                print(f'Train Epoch {epoch}.{i}: loss={loss}')
                del train_loader
                gc.collect()
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

if __name__ == '__main__':
    run()

