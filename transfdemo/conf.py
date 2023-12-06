import torch


d_model = 6  # embedding size
# d_model = 3  # embedding size
d_ff = 12  # feedforward nerual network  dimension
d_k = d_v = 3  # dimension of k(same as q) and v
n_heads = 2  # number of heads in multihead attention
# n_heads = 1  # number of heads in multihead attention【注：为debug更简单，可以先改为1个head】
p_drop = 0.1  # propability of dropout
device = "cpu"

# 定义词典
source_vocab = {'E': 0, '我': 1, '吃': 2, '肉': 3}
target_vocab = {'E': 0, 'I': 1, 'eat': 2, 'meat': 3, 'S': 4}

# 样本数据
encoder_input = torch.LongTensor([[1, 2, 3, 0]]).to(device)  # 我 吃 肉 E, E代表结束词
decoder_input = torch.LongTensor([[4, 1, 2, 3]]).to(device)  # S I eat meat, S代表开始词, 并右移一位，用于并行训练
target = torch.LongTensor([[1, 2, 3, 0]]).to(device)  # I eat meat E, 翻译目标

MODEL_PATH = 'model.pt'