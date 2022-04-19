#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchviz import make_dot

# In[2]:


num_epoch = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# In[3]:


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


# In[4]:


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


dummy_x = torch.randint(0, 128, (batch_size_train, 1, 28, 28)) * 1.0
dummy_x.requires_grad = True
dummy_y = torch.randint(0, 10, (batch_size_train,))
dummy_set = ImgDataset(dummy_x, dummy_y)
dummy_loader = DataLoader(dummy_set, batch_size=batch_size_train, shuffle=True)


# In[5]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer0 = nn.Conv2d(1, 10, 5, 1, 1)
        self.layer1 = nn.MaxPool2d(2, 2, 0)
        self.layer2 = nn.Conv2d(10, 20, 5, 1, 1)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d(2, 2, 0)
        self.layer5 = nn.Linear(20 * 5 * 5, 100)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size()[0], -1)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out


# In[6]:


model = Classifier()
model_inv = Classifier()


# In[7]:


def model_copy(src, dst):
    for slayer in src.named_children():
        if 'weight' not in dir(slayer[1]):
            continue
        for dlayer in dst.named_children():
            if 'weight' not in dir(dlayer[1]):
                continue
            if slayer[0] == dlayer[0]:
                dlayer[1].weight.data = slayer[1].weight.data.clone()
                break


# In[8]:


model = Classifier().cuda()
model_target = Classifier().cuda()
model_inv = Classifier().cuda()
model_copy(model, model_inv)
model_copy(model, model_target)
torch.cuda.empty_cache()
print(model.layer0.weight.data[0, 0, 0, :5])
print(model_inv.layer0.weight.data[0, 0, 0, :5])
print(model.layer0.weight.grad)
print(model_inv.layer0.weight.grad)

# In[9]:


modelx = model_target.cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.SGD(modelx.parameters(), lr=learning_rate)  # optimizer 使用 SGD
num_epoch = 1
begin = time.time()
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    test_acc = 0.0
    test_loss = 0.0

    modelx.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = modelx(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    modelx.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = modelx(data[0].cuda())
            batch_loss = loss(test_pred, data[1].cuda())

            test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            test_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Test Acc: %3.6f loss: %3.6f' % (
        epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / train_loader.dataset.__len__(),
        train_loss / train_loader.dataset.__len__(), test_acc / test_loader.dataset.__len__(),
        test_loss / test_loader.dataset.__len__()))
print("total %2.2f sec(s)" % (time.time() - begin))


# In[10]:


def weightLoss(src, dst):
    loss = torch.tensor(0.0, requires_grad=True)
    for slayer in src.named_children():
        if 'weight' not in dir(slayer[1]):
            continue
        for dlayer in dst.named_children():
            if 'weight' not in dir(dlayer[1]):
                continue
            if slayer[0] == dlayer[0]:
                loss = loss + sum(((slayer[1].weight.cuda() - dlayer[1].weight.cuda()) ** 2).view(-1))
                break
    return loss


def stepWeight(weight):
    # print(weight.grad[:5])
    weight = weight + learning_rate * weight.grad


# In[11]:


model_copy(model, model_inv)
modelx = model_inv.cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.SGD(modelx.parameters(), lr=learning_rate)  # optimizer 使用 SGD
num_epoch = 1
begin = time.time()
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    test_acc = 0.0
    test_loss = 0.0

    modelx.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(dummy_loader):
        data[0] = nn.Parameter(data[0])
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = modelx(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        data[0].grad.zero_()
        wloss = weightLoss(modelx, model_target)
        wloss.backward()
        stepWeight(data[0])
        data[0].grad.zero_()

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % (
    epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / dummy_loader.dataset.__len__(),
    train_loss / dummy_loader.dataset.__len__(),))
print("total %2.2f sec(s)" % (time.time() - begin))

