import os
import torch
import tqdm
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchnet import meter

class CaptchaNet(nn.Module):
    '''
    '''

    def __init__(self):
        '''
        '''
        
        super().__init__()

        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.conv3 = nn.Conv2d(10, 16, 6)

        # 线性
        self.fc1 = nn.Linear(4 * 12 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # 验证码 5 个，10 是字符集个数
        self.fc41 = nn.Linear(128, 10)
        self.fc42 = nn.Linear(128, 10)
        self.fc43 = nn.Linear(128, 10)
        self.fc44 = nn.Linear(128, 10)
        self.fc45 = nn.Linear(128, 10)

    def forward(self, x):
        '''
        '''
        # => (3, 128, 64)
        # print(f'x 0 size: {x.size()}')
        
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
        # => (5, 62, 30)
        # print(f'x 1 size: {x.size()}')

        x = torch.max_pool2d(torch.relu(self.conv2(x)), (2, 2))
        # => (10, 29, 13)
        # print(f'x 2 size: {x.size()}')

        x = torch.max_pool2d(torch.relu(self.conv3(x)), (2, 2))
        # => (16, 12, 4)
        # print(f'x 3 size: {x.size()}')

        x = x.view(-1, 768) # 数据扁平化
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)

        x1 = self.fc41(x)
        x2 = self.fc42(x)
        x3 = self.fc43(x)
        x4 = self.fc44(x)
        x5 = self.fc45(x)

        return x1, x2, x3, x4, x5

    def answer(self, x):
        '''
        '''

        y1, y2, y3, y4, y5 = self(x)
        y1 = y1.topk(1, dim=1)[1]
        y2 = y2.topk(1, dim=1)[1]
        y3 = y3.topk(1, dim=1)[1]
        y4 = y4.topk(1, dim=1)[1]
        y5 = y5.topk(1, dim=1)[1]
        return torch.cat((y1, y2, y3, y4, y5), dim=1)


    def load(self, path):
        '''
        '''

        if os.path.isfile(path):
            d = torch.load(path)
            self.load_state_dict(d)

    def save(self, path):
        '''
        '''
        
        d = self.state_dict()
        torch.save(d, path)

class CaptchaTrainer:
    '''
    '''
    
    def __init__(self, mpath, learningRate=1e-3):
        '''
        '''

        self.net = CaptchaNet()
        self.mpath = mpath
        if os.path.isfile(mpath):
            self.net.load(mpath)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate)
        self.loss_meter = meter.AverageValueMeter()
    
    def train(self, train_dataset, test_dataset, epoch_count=200, batch_size=128, save_circle=200, print_circle=100, test_circle=100):
        '''
        '''

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        avg_loss = 0.0
        for epoch in range(epoch_count):
            for i, (x, label) in tqdm.tqdm(enumerate(train_data_loader, 0)):
                label = label.long()
                l1, l2, l3, l4, l5 = label[:,0], label[:,1], label[:,2], label[:,3],  label[:,4]
                self.optimizer.zero_grad()
                y1, y2, y3, y4, y5 = self.net(x)
                loss1 = self.criterion(y1, l1)
                loss2 = self.criterion(y2, l2)
                loss3 = self.criterion(y3, l3)
                loss4 = self.criterion(y4, l4)
                loss5 = self.criterion(y5, l5)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                self.loss_meter.add(loss.item())
                avg_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if i % print_circle == 1:
                    print(f'{i} loss: {avg_loss / print_circle}')
                    avg_loss = 0

                if i % test_circle == 1:
                    accuracy = self.test(test_data_loader, batch_size)
                    print(f'accuracy: {accuracy}')

                if i % save_circle == 1:
                    self.net.save(self.mpath)
                    print(f'save: {i}')

    def test(self, test_data_loader, batch_size, test_num=6):
        '''
        '''

        total_num = test_num * batch_size
        right_num = 0

        for i, (x, label) in enumerate(test_data_loader, 0):
            if i >= test_num:
                break

            label = label.long()
            y1, y2, y3, y4, y5 = self.net(x)
            y1 = y1.topk(1, dim=1)[1].view(batch_size, 1)
            y2 = y2.topk(1, dim=1)[1].view(batch_size, 1)
            y3 = y3.topk(1, dim=1)[1].view(batch_size, 1)
            y4 = y4.topk(1, dim=1)[1].view(batch_size, 1)
            y5 = y5.topk(1, dim=1)[1].view(batch_size, 1)
            y = torch.cat((y1, y2, y3, y4, y5), dim=1)
            diff = (y != label)
            diff = diff.sum(1)
            diff = (diff != 0)
            res = diff.sum(0).item()
            right_num += (batch_size - res)
        
        accuracy = float(right_num) / float(total_num)
        return accuracy
