import numpy as np
import torch
import torch.nn as nn
from conf import * 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q、K、V，此时是已经乘过 W(q)、W(k)、W(v) 矩阵
        # 如下图，但不用一个个算，矩阵乘法一次搞定
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 遮盖区的值设为近0，表示E结尾 or decoder 自我顺序遮盖，注意力丢弃
        scores.masked_fill_(attn_mask, -1e9)
        # softmax后（遮盖区变为0）
        attn = nn.Softmax(dim=-1)(scores)
        # 乘积意义：给V带上了注意力信息。prob就是下图z（矩阵计算不用在v1+v2）。
        prob = torch.matmul(attn, V)
        return prob
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.sdpa = ScaledDotProductAttention()
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)  # ff 全连接
        self.layer_norm = nn.LayerNorm(d_model)  # normal 归一化

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q：1*4*6，每批1句 * 每句4个词 * 每词6长度编码
        # residual 先临时保存下：原始值，后面做残差连接加法
        residual, batch = input_Q, input_Q.size(0)

        # 乘上 W 矩阵。注：W 就是要训练的参数
        # 注意：维度从2维变成3维，增加 head 维度，也是一次性并行计算
        Q = self.W_Q(input_Q)  # 乘以 W(6*6) 变为 1*4*6
        Q = Q.view(batch, -1, n_heads, d_k).transpose(1, 2)  # 切开为2个Head 变为 1*2*4*3 1批 2个Head 4词 3编码
        K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2)

        # 1*2*4*4，2个Head的4*4，最后一列为true
        # 因为最后一列是 E 结束符
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 返回1*2*4*3，2个头，4*3为带上关注关系的4词
        prob = self.sdpa(Q, K, V, attn_mask)

        # 把2头重新拼接起来，变为 1*4*6
        prob = prob.transpose(1, 2).contiguous()
        prob = prob.view(batch, -1, n_heads * d_v).contiguous()

        # 全连接层：对多头注意力的输出进行线性变换，从而更好地提取信息
        output = self.fc(prob)

        # 残差连接 & 归一化
        res = self.layer_norm(residual + output) # return 1*4*6
        return res
    
def get_attn_pad_mask(seq_q, seq_k):  # 本质是结尾E做注意力遮盖，返回 1*4*4，最后一列为True
    batch, len_q = seq_q.size()  # 1, 4
    batch, len_k = seq_k.size()  # 1, 4
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 为0则为true，变为f,f,f,true，意思是把0这个结尾标志为true
    return pad_attn_mask.expand(batch, len_q, len_k)  # 扩展为1*4*4，最后一列为true，表示抹掉结尾对应的注意力


def get_attn_subsequent_mask(seq):  # decoder的自我顺序注意力遮盖，右上三角形区为true的遮盖
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(len(source_vocab), d_model)
        self.attention = MultiHeadAttention()

    def forward(self, encoder_input):
        # input 1 * 4，1句话4个单词
        # 1 * 4 * 6，将每个单词的整数字编码扩展到6个浮点数编码
        embedded = self.source_embedding(encoder_input)
        # 1 * 4 * 4 矩阵，最后一列为true，表示忽略结尾词的注意力机制
        mask = get_attn_pad_mask(encoder_input, encoder_input)
        # 1*4*6，带上关注力的4个词矩阵
        encoder_output = self.attention(embedded, embedded, embedded, mask)
        return encoder_output
    
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(len(target_vocab), d_model)
        self.attention = MultiHeadAttention() # 只有1级

    # 三入参形状分别为 1*4, 1*4, 1*4*6，前两者未被embedding，注意后面这个是 encoder_output
    def forward(self, decoder_input, encoder_input, encoder_output):
        # 编码为1*4*6
        decoder_embedded = self.target_embedding(decoder_input)

        # 1*4*4 全为false，表示没有结尾词
        decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input)
        # 1*4*4 右上三角区为1，其余为0
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input)
        # 1*4*4 右上三角区为true，其余为false
        decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0)

        # 1*4*6 带上注意力的4词矩阵【注：decoder里面，第1个attention】
        decoder_output = self.attention(decoder_embedded, decoder_embedded, decoder_embedded, decoder_self_mask)

        # 1*4*4 最后一列为true，表示E结尾词
        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input)
        # 输入均为 1*4*6，Q表示"S I eat meat"、K表示"我吃肉E"、V表示 "我吃肉E"
        #【注：decoder里面，第2个attention】
        decoder_output = self.attention(decoder_output, encoder_output, encoder_output, decoder_encoder_attn_mask)
        return decoder_output
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(d_model, len(target_vocab), bias=False)

    def forward(self, encoder_input, decoder_input):
        # 编码
        encoder_output = self.encoder(encoder_input)
        # 解码
        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
        
        # 预测出4个词，每个词对应到词典中5个词的概率，如下
        # tensor([[[ 0.0755, -0.2646,  0.1279, -0.3735, -0.2351],[-1.2789,  0.6237, -0.6452,  1.1632,  0.6479]]]
        decoder_logits = self.fc(decoder_output)
        res = decoder_logits.view(-1, decoder_logits.size(-1))
        return res
    
class Subare(nn.Module):
    def __init__(self):
        super(Subare, self).__init__()
        self.lv1 = Transformer()
        self.lv2 = Transformer()

    def forward(self, encoder_input, decoder_input):
        lv1_output = self.lv1(encoder_input, decoder_input)
