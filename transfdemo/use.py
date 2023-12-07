import os
from model import Transformer
from conf import *

model = Transformer().to(device)

if os.path.isfile(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(state)

# 预测目标是5个单词
target_len = len(target_vocab)  
# 1*4*6 输入"我吃肉E"，先算【自注意力】
encoder_output = model.encoder(encoder_input)  
# 1*5 全是0，表示EEEEE
decoder_input = torch.zeros(1, target_len).type_as(encoder_input.data)  


# S 开始输出
next_symbol = SYMBOL_START
# 5个单词逐个预测【注意：是一个个追加词，不断往后预测的】
for i in range(target_len):  
    # 譬如i=0第一轮，decoder输入为SEEEE，第二轮为S I EEE，把预测 I 给拼上去，继续循环
    decoder_input[0][i] = next_symbol

    # decoder 输出
    decoder_output = model.decoder(decoder_input, encoder_input, encoder_output)
    # 负责将解码器的输出映射到目标词汇表，每个元素表示对应目标词汇的分数
    # 取出最大的五个词的下标，譬如[1, 3, 3, 3, 3] 表示 i,meat,meat,meat,meat
    logits = model.fc(decoder_output).squeeze(0)
    prob = logits.max(dim=1, keepdim=False)[1]
    next_symbol = prob.data[i].item()  # 只取当前i

    for k, v in target_vocab.items():
        if v == next_symbol:
            print('第', i, '轮:', k)
            break

    if next_symbol == SYMBOL_END:  # 遇到结尾了，那就完成翻译
        break