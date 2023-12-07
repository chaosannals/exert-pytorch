import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import Transformer
from conf import *

model = Transformer().to(device)

if os.path.isfile(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(state)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)

for epoch in range(10):
    # 训练数据就 1 条， TODO 改用数据集
    output = model(encoder_input, decoder_input)

    # print(f'encoder_input: {encoder_input}')
    # print(f'output: {output}')

    # 和目标词 I eat meat E做差异计算
    loss = criterion(output, target.view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 保存模型
torch.save(model.state_dict(), 'model.pt')