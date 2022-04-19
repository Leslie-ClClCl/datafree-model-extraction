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
target_model.load_state_dict(torch.load('./target_model/best.pt'))
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
#     # eval
#     target_model.eval()
#     correct = 0
#     total = 0
#     for idx, (image, label) in enumerate(cifar_test_loader):
#         image, label = image.cuda(), label.cuda()
#         pred = target_model(image)
#         _, pred = torch.max(pred, dim=1)
#         total += label.shape[0]
#         correct += (pred == label).sum().item()
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


class simple_mlp(nn.Module):
    def __init__(self):
        super(simple_mlp, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, input):
        relu = nn.ReLU()
        softmax = nn.Softmax()
        output = relu(self.fc1(input))
        output = relu(self.fc2(output))
        output = softmax(self.fc3(output))
        return output


# print('resnet 18 acc: {}'.format(evaluate(target_model, attack_train_img, attack_train_class)))

# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# attack_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
# attack_model.fit(attack_train_X, attack_train_y)

# attack_model = lgb.LGBMClassifier(objective='binary', reg_lambda=0.2, n_estimators=10000, learning_rate=0.1)
# attack_model = svm.SVC(kernel='rbf', verbose=True)
# attack_model.fit(attack_train_X, attack_train_y)
attack_model = simple_mlp().cuda()
# attack_model.load_state_dict(torch.load('./attack_model/best.pt'))
# train attack model
criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(attack_model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
attack_model.train()
for i in range(200):
    loss_e = 0
    for idx, (X, y) in enumerate(attack_train_loader):
        X, y = X.cuda(), y.cuda()
        target_pred = target_model(X)
        target_pred, _ = torch.topk(target_pred, k=10, dim=1)
        optimizer.zero_grad()
        attack_output = attack_model(target_pred)
        loss = criteria(attack_output, y)
        loss.backward()
        optimizer.step()
        loss_e += loss.item()
    scheduler.step()
    print('epoch {} loss {}'.format(i, loss_e))
torch.save(attack_model.state_dict(), './attack_model/best.pt')
# evaluate attack model
attack_model.eval()
correct = 0
total = 0
for idx, (X, y) in enumerate(attack_test_loader):
    X, y = X.cuda(), y.cuda()
    target_output = target_model(X, logits=True, temperature=2)
    target_output, _ = torch.topk(target_output, k=10, dim=1)
    pred = attack_model(target_output)
    pred = torch.argmax(pred, dim=1)
    correct += torch.eq(pred, y).sum().item()
    total += y.shape[0]
print('acc {}'.format(correct/total))
