import lightgbm as lgb
import os
from sklearn import svm
import np as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchvision.transforms import transforms

from shadow_data import split_cifar_for_attack, get_average_cifar10
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from resnet import resnet18
import numpy as np

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('rm tmp')


target_model = resnet18(10).cuda()
target_model.load_state_dict(torch.load('./target_model/cifar10_resnet18_student.pt'))
cifar_train_avg, cifar_test_avg = get_average_cifar10()
cifar_train_loader, cifar_test_loader = data.DataLoader(cifar_train_avg, batch_size=128, shuffle=True), \
                                        data.DataLoader(cifar_test_avg, batch_size=128, shuffle=False)

# # Train Target Model
# criteria = nn.CrossEntropyLoss()
# optimizer = optim.SGD(target_model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180], 0.1)
# best_acc = 0.0
# for epoch in range(200):
#     loss_e = 0
#     for idx, (image, label) in enumerate(cifar_train_loader):
#         optimizer.zero_grad()
#         target_model.train()
#         image, label = image.cuda(), label.cuda()
#         pred = target_model(image)
#         loss = criteria(pred, label)
#         loss_e += loss.item()
#         loss.backward()
#         optimizer.step()
#     scheduler.step()
#
# eval
target_model.eval()
correct = 0
total = 0
for idx, (image, label) in enumerate(cifar_test_loader):
    image, label = image.cuda(), label.cuda()
    pred = target_model(image)
    _, pred = torch.max(pred, dim=1)
    total += label.shape[0]
    correct += (pred == label).sum().item()
print('acc {}'.format(correct/total))
#     if best_acc < correct/total:
#         best_acc = correct/total
#         torch.save(target_model.state_dict(), './target_model/best.pt')
#     print('epoch {} loss {} acc {}'.format(epoch, loss_e, correct/total))

# target_model.load_state_dict(checkpoint)
# 划分得到用于训练attack model的数据
attack_train, attack_test = split_cifar_for_attack(cifar_train_avg, cifar_test_avg)
attack_train_loader = data.DataLoader(attack_train, batch_size=256, shuffle=True)
attack_test_loader = data.DataLoader(attack_test, batch_size=256, shuffle=False)


def evaluate(model, input, true_label):
    model.eval()
    pred = model(input)
    pred = pred.argmax(dim=1)
    correct = torch.eq(pred, true_label).sum().float().item()
    acc = correct/18000
    return acc


class TpMLP(nn.Module):
    def __init__(self):
        super(TpMLP, self).__init__()
        self.p1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, fx, label):
        output1 = self.p1(fx)
        output2 = self.p2(label)
        output = self.p3(torch.concat((output1, output2), dim=1))
        return output


def get_onehot_label(labels, num_total=10):
    arr = torch.zeros((labels.shape[0], num_total))
    arr[np.arange(0, labels.shape[0]), labels] = 1
    return arr.cuda()


attack_model = TpMLP().cuda()
# attack_model.load_state_dict(torch.load('./attack_model/best.pt'))

# train attack model
criteria = nn.MSELoss()
optimizer = optim.SGD(attack_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for i in range(200):
    loss_e = 0
    samples_num = 0
    attack_model.train()
    for idx, (X, label, y) in enumerate(attack_train_loader):
        optimizer.zero_grad()
        X, y = X.cuda(), y.cuda()
        label = torch.tensor(label, dtype=torch.float32).cuda()
        label = label.unsqueeze(1)
        fx = target_model(X)

        attack_output = attack_model(fx, get_onehot_label(y))
        loss = criteria(attack_output, label)
        loss.backward()
        optimizer.step()
        loss_e += loss.item()
        samples_num += label.shape[0]

    scheduler.step()
    print('epoch {} loss {}'.format(i, loss_e / samples_num))
    # evaluate attack model
    attack_model.eval()
    correct = 0
    total = 0
    for idx, (X, label, y) in enumerate(attack_test_loader):
        X, y, label = X.cuda(), y.cuda(), label.cuda()
        label = label.unsqueeze(1)
        fx = target_model(X)

        attack_pred = attack_model(fx, get_onehot_label(y))
        attack_pred = torch.where(attack_pred > 0.5, 1, 0)
        correct += torch.eq(attack_pred, label).sum().item()
        total += y.shape[0]
    print('acc {}'.format(correct / total))
torch.save(attack_model.state_dict(), './attack_model/best.pt')
