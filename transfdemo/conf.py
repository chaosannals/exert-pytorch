import torch

# 字典维度，调参时要根据训练资料量（训练资料总体词汇量）调整。
# 越大表现力越强（词汇量大），但是训练难度越大，模型越大。
# 越小表现力越弱（词汇量小），但是训练难度越小，模型越小。
d_model = 10

d_ff = 12  # feedforward nerual network  dimension
d_k = d_v = 3  # dimension of k(same as q) and v
n_heads = 2  # number of heads in multihead attention

p_drop = 0.1  # propability of dropout
device = "cpu"

# 定义词典
SYMBOL_END = 0
SYMBOL_START = 1 

# TODO 统一用同个字典
source_vocab = {'E': SYMBOL_END, 'S': SYMBOL_START, '我': 2, '吃': 3, '肉': 4}
target_vocab = {'E': SYMBOL_END, 'S': SYMBOL_START, '你': 2, '放': 3, '屁': 4}

# TODO 样本数据，改用数据集
encoder_input = torch.LongTensor([[2, 3, 4, SYMBOL_END]]).to(device)  # E代表结束词
decoder_input = torch.LongTensor([[SYMBOL_START, 2, 3, 4]]).to(device)  # S代表开始词, 并右移一位，用于并行训练
target = torch.LongTensor([[2, 3, 4, SYMBOL_END]]).to(device)  #  下一句回答的目标

MODEL_PATH = 'model.pt'