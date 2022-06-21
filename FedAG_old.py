import argparse
import copy
import importlib
import os
import threading
from datetime import timezone, timedelta, datetime

import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from models import Logistic, TwoHiddenLayerFc, TwoConvOneFc, CifarCnn, LeNet, Generator

tz_utc_8 = timezone(timedelta(hours=8))

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('rm tmp')


def read_data(path: str):
    res = []
    res.append(torch.load(os.path.join(path, 'train/train.pt')))
    res.append(torch.load(os.path.join(path, 'test/test.pt')))
    users = res[0]['users']
    return users, res[0]['user_data'], res[1]['user_data'], res[0]['num_samples'], res[1]['num_samples']


def train_on_local_data(model, train_dataloader, optimizer, scheduler, criteria, epoch, cuda_available=True):
    """ 客户端模型在本地数据集上的训练"""
    model.cuda()
    loss_total = []
    for e_idx in range(epoch):
        loss_e = 0.0
        for X, y in train_dataloader:
            if cuda_available:
                X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(X)
            loss = criteria(pred, y)
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
        loss_total.append(loss_e)
        if scheduler is not None:
            scheduler.step()
    model.cpu()
    return loss_total


def distill(model, sample_loader, optimizer, scheduler, criteria, epoch, cuda_available=True):
    """ 蒸馏训练过程 """
    model.cuda()
    loss_total = []
    for e_idx in range(epoch):
        loss_e = 0.0
        for X, y in sample_loader:
            if cuda_available:
                X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(X)
            # TODO 输出结果要不要经过softmax
            loss = criteria(pred, y)
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
        loss_total.append(loss_e)
        if scheduler is not None:
            scheduler.step()
    model.cpu()
    return loss_total


def train_generator(generator, client_model, server_model, optimizer_G, optimizer_C, local_epoch, batch_size, nz, criteria):
    generator.cuda(), client_model.cuda(), server_model.cuda()
    for i in range(local_epoch):
        if optimizer_C is not None:
            for k in range(5):
                z = torch.randn((batch_size, nz, 1, 1)).cuda()
                optimizer_C.zero_grad()
                fake = generator(z).detach()
                t_logit = server_model(fake)
                s_logit = client_model(fake)
                loss_C = criteria(s_logit, t_logit.detach())

                loss_C.backward()
                optimizer_C.step()

        z = torch.randn((batch_size, nz, 1, 1)).cuda()
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        t_logit = server_model(fake)
        s_logit = client_model(fake)

        loss_G = - criteria(s_logit, t_logit)
        loss_G.backward()
        optimizer_G.step()
    generator.cpu(), client_model.cpu(), server_model.cpu()
    return loss_G.item()


def param_avg(models, avg_model):
    """ 聚合模型参数 """
    model_num = len(models)
    assert model_num > 0
    # 模型参数扁平化
    flatten_models = []
    for model in models:
        flatten_params = []
        for params in model.parameters():
            flatten_params.append(params.data.view(-1))
        flatten_models.append(torch.cat(flatten_params).detach())
    flatten_avg_model = torch.zeros_like(flatten_models[0])
    for flatten_model in flatten_models:
        flatten_avg_model += flatten_model
    flatten_avg_model /= model_num
    prev_idx = 0
    for params in avg_model.parameters():
        flat_size = int(np.prod(list(params.size())))
        params.data.copy_(
            flatten_avg_model[prev_idx:prev_idx + flat_size].view(params.size()))
        prev_idx += flat_size


def get_noise_samples(generator, sample_num, nz):
    """ 生成样本， 样本数量是100的倍数 """
    generator.cuda()
    fake_img = []
    for g_idx in range(int(sample_num/100)):
        z = torch.randn((100, nz, 1, 1)).cuda()
        fake_img_b = generator(z).detach()
        fake_img.extend(fake_img_b[i] for i in range(100))
    fake_img = torch.stack(fake_img)
    generator.cpu()
    return fake_img.cpu()


def get_model(model_name, input_shape, num_class):
    if model_name == 'logistic':
        return Logistic(input_shape, num_class)
    elif model_name == '2nn':
        return TwoHiddenLayerFc(input_shape, num_class)
    elif model_name == 'cnn':
        return TwoConvOneFc(input_shape, num_class)
    elif model_name == 'ccnn':
        return CifarCnn(input_shape, num_class)
    elif model_name == 'lenet':
        return LeNet(input_shape, num_class)
    elif model_name.startswith('vgg'):
        mod = importlib.import_module('src.models.vgg')
        vgg_model = getattr(mod, model_name)
        return vgg_model(num_class)
    else:
        raise ValueError("Not support model: {}!".format(model_name))


def get_generator(nz=100, ngf=64, nc=1, img_size=32):
    return Generator(nz, ngf, nc, img_size)


def select_users(users, users_per_round, seed=1):
    """ 按照设置的每轮参与训练的用户数量，随机挑选客户端 """
    num_users = min(users_per_round, len(users))
    np.random.seed(seed)
    return np.random.choice(users, num_users, replace=False).tolist()


class MiniDataset(Dataset):
    def __init__(self, data_list, transforms):
        super(MiniDataset, self).__init__()
        assert len(data_list) == len(transforms)
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, idx):
        res = []
        for data_idx, data in enumerate(self.data_list):
            if self.transforms[data_idx] is not None:
                res.append(self.transforms[data_idx](data[idx]))
            else:
                res.append(data[idx])
        return res

    def __len__(self):
        return self.data_list[0].shape[0]


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', help='name of trainer;', type=str, default='fedad')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--comm_round', type=int, default=200)
    parser.add_argument('--num_user', help='number of users in total', type=int, default=20)
    parser.add_argument('--users_per_round', type=int, default=10)
    # dataset parameters
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='Mnist')
    parser.add_argument('--category', help='number of dataset category, 10 for MNIST', type=int, default=10)
    parser.add_argument('--alpha', help='alpha in Dirichelt distribution (smaller means larger heterogeneity)',
                        type=int, default=0.4)
    parser.add_argument("--sampling_ratio", type=float, default=0.25, help="Ratio for sampling training samples.")
    parser.add_argument("--batch_size", type=int, default=64)
    # trainer parameters
    parser.add_argument('--lr_C_local_train', type=float, default=1e-2)
    parser.add_argument('--lr_G_local', type=float, default=1e-4)
    parser.add_argument('--lr_C_global_distill', type=float, default=1e-3)
    parser.add_argument('--lr_C_local_distill', type=float, default=1e-4)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--global_dis_epoch', type=int, default=50)
    # generator parameters
    parser.add_argument('--noise_batch', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--sample_per_round', type=int, default=10000)
    parsed = parser.parse_args()
    options = parsed.__dict__
    for key, value in options.items():
        print('{}\t:{}'.format(key, value))
    return options


def get_prediction(model, samples):
    model.cuda()
    dataloader = DataLoader(MiniDataset([samples], [None]), batch_size=128)
    res = []
    for X in dataloader:
        X = X[0].cuda()
        pred = model(X)
        res.extend(pred[i] for i in range(pred.shape[0]))
    res = torch.stack(res)
    model.cpu()
    return res.detach().cpu()


def main():
    options = read_options()
    non_iid_fold = os.path.join('data', options['dataset']+'/u{}c{}-alpha{}-ratio{}'.format(options['num_user'],
                                                                                            options['category'],
                                                                                            options['alpha'],
                                                                                            options['sampling_ratio']))
    suffix = '_'.join(map(str, [arg for arg in options.values()])) + datetime.now(tz_utc_8).strftime("_%m-%d_%H:%M:%S")
    if not os.path.exists('result/'+suffix):
        os.mkdir('result/'+suffix)
    users, train_data, test_data, num_train_data, num_test_data = read_data(non_iid_fold)
    train_dataloader = {user: DataLoader(MiniDataset(list(train_data[user].values()), [None, None]),
                                         batch_size=options['batch_size']) for user in users}
    test_dataloader = {user: DataLoader(MiniDataset(list(test_data[user].values()), [None, None]),
                                        batch_size=options['batch_size']) for user in users}
    train_writer = SummaryWriter(os.path.join('result', suffix+'/train/'))
    test_writer = SummaryWriter(os.path.join('result', suffix+'/test/'))
    # 全局模型和优化器
    global_model = get_model(options['model'], (1, 28, 28), options['category'])
    global_generator = get_generator(nz=options['nz'], nc=1, img_size=28)
    global_optim = torch.optim.RMSprop(global_model.parameters(), lr=options['lr_C_global_distill'])
    # 客户端模型、生成器、优化器
    user_models = {}
    user_generator = {}
    user_optim_C = {}
    user_optim_dis = {}
    user_optim_G = {}
    for user in users:
        user_models[user] = get_model(options['model'], (1, 28, 28), options['category'])
        user_generator[user] = get_generator(nz=options['nz'], nc=1, img_size=28)

        user_models[user].load_state_dict(copy.deepcopy(global_model.state_dict()))
        user_generator[user].load_state_dict(copy.deepcopy(global_generator.state_dict()))
        user_optim_C[user] = torch.optim.SGD(user_models[user].parameters(), lr=options['lr_C_local_train'],
                                             momentum=0.9, weight_decay=5e-4)
        user_optim_dis[user] = torch.optim.Adam(user_models[user].parameters(), lr=options['lr_C_local_distill'])
        user_optim_G[user] = torch.optim.SGD(user_generator[user].parameters(), lr=options['lr_G_local'])
    # 损失计算方法
    criteria_ce = torch.nn.CrossEntropyLoss()
    criteria_l1 = torch.nn.L1Loss()
    criteria_KL = torch.nn.KLDivLoss()

    # 联邦训练的过程
    # 预训练过程
    for user in users:
        train_on_local_data(user_models[user], train_dataloader[user], user_optim_C[user], None, criteria_ce,
                            options['local_epoch'])

    best_acc = 0.0
    # online训练过程
    for round_idx in range(options['comm_round']):
        selected_users = select_users(users, options['users_per_round'], seed=round_idx)
        # 客户度本地更新生成器
        for user in selected_users:
            train_on_local_data(user_models[user], train_dataloader[user], user_optim_C[user], None, criteria_ce,
                                options['local_epoch'])
            loss_G = train_generator(user_generator[user], user_models[user], global_model, user_optim_G[user],
                                     None, local_epoch=options['local_epoch'],
                                     batch_size=options['noise_batch'], nz=options['nz'], criteria=criteria_l1)
            train_writer.add_scalar('{} generator training loss'.format(user), loss_G, round_idx)
        # 聚合生成器
        param_avg([user_generator[user] for user in selected_users], global_generator)
        for user in selected_users:
            user_generator[user].load_state_dict(copy.deepcopy(global_generator.state_dict()))
        # 生成器产生公共数据
        samples_gen = get_noise_samples(global_generator, options['sample_per_round'], options['nz'])
        # 客户端输出预测分数
        prediction_avg = torch.zeros((samples_gen.shape[0], options['category']))
        for user in selected_users:
            prediction_avg += get_prediction(user_models[user], samples_gen)
        prediction_avg /= len(selected_users)
        # 全局模型的蒸馏过程
        sample_pred_loader = DataLoader(MiniDataset((samples_gen, prediction_avg.detach()), [None, None]), batch_size=64)
        distill(global_model, sample_pred_loader, global_optim, None, criteria_l1, options['global_dis_epoch'])
        prediction_global = get_prediction(global_model, samples_gen)
        sample_pred_loader = DataLoader(MiniDataset((samples_gen, prediction_global.detach()), [None, None]), batch_size=64)
        # 客户端的本地更新
        for user in selected_users:
            loss_dis = distill(user_models[user], sample_pred_loader, user_optim_dis[user], None, criteria_l1,
                               options['local_epoch'])
            # TODO 记录loss
        # 测试精度
        total = 0
        correct = 0
        for user in users:
            for X, y in test_dataloader[user]:
                pred = global_model(X)
                pred = torch.argmax(pred, dim=1)
                correct += torch.eq(pred, y).sum().item()
                total += y.shape[0]
        print('round {}, global model accuracy {}'.format(round_idx, correct/total))
        test_writer.add_scalar('global model acc', correct/total, round_idx)


if __name__ == "__main__":
    main()
