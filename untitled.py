import argparse
import copy
import importlib
import os
import random
import time
from datetime import timezone, timedelta, datetime

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models import Logistic, TwoHiddenLayerFc, TwoConvOneFc, CifarCnn, LeNet, Generator
from FedGen.FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

tz_utc_8 = timezone(timedelta(hours=8))

random_int = random.randint(0, 10000)
time.sleep(random_int / 500)
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp{}'.format(random_int))
memory_gpu = [int(x.split()[2]) for x in open('./tmp{}'.format(random_int), 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.system('rm tmp{}'.format(random_int))

criterias = {'l1': torch.nn.L1Loss(),
             'l2': torch.nn.MSELoss(),
             'kl': torch.nn.KLDivLoss()}
optimizers = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'RMSprop': torch.optim.RMSprop}


def read_data(path: str):
    """ 读取数据 """
    res = []
    res.append(torch.load(os.path.join(path, 'train/train.pt')))
    res.append(torch.load(os.path.join(path, 'test/test.pt')))
    users = res[0]['users']
    return users, res[0]['user_data'], res[1]['user_data'], res[0]['num_samples'], res[1]['num_samples']


def train_on_local_data(model, train_dataloader, optimizer, scheduler, criteria, steps, cuda_available=True):
    """ 客户端模型在本地数据集上的训练 """
    loss_total = []
    train_dataiter = iter(train_dataloader)
    for e_idx in range(steps):
        try:
            X, y = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            X, y = next(train_dataiter)
        if cuda_available:
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = model(X)

        if torch.isnan(pred.max()):
            from IPython import embed
            embed()

        loss = criteria(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 60)
        optimizer.step()
        loss_total.append(loss.item())
    if scheduler is not None:
        scheduler.step()
    return loss_total


def distill(model, sample_loader, optimizer, scheduler, epoch, loss_type='l1', temperature=3, cuda_available=True):
    """ 蒸馏训练过程 """
    loss_total = []
    sample_iter = iter(sample_loader)
    if loss_type == 'l1':
        criteria = torch.nn.L1Loss()
    elif loss_type == 'l2':
        criteria = torch.nn.MSELoss()
    elif loss_type == 'kl':
        criteria = torch.nn.KLDivLoss()

    for e_idx in range(epoch):
        try:
            X, y = next(sample_iter)
        except StopIteration:
            sample_iter = iter(sample_loader)
            X, y = next(sample_iter)
        if cuda_available:
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = model(X)
        if loss_type == 'kl':
            pred = F.log_softmax(pred / temperature)
            y = F.softmax(y / temperature)
        else:
            pred = F.softmax(pred / temperature)
            y = F.softmax(y / temperature)
        loss = criteria(pred, y)
        loss.backward()
        optimizer.step()
        loss_total.append(loss.item())
    if scheduler is not None:
        scheduler.step()
    return loss_total


def train_generator(generator, client_model, server_model, optimizer_G, scheduler_G, local_epoch,
                    batch_size, nz, criteria):
    """ 利用本地模型和全局模型 训练生成器 """
    for i in range(local_epoch):
        optimizer_G.zero_grad()
        generator.train()

        z = torch.randn((batch_size, nz, 1, 1)).cuda()
        fake = generator(z)
        t_logit = server_model(fake)
        s_logit = client_model(fake)

        loss_G = - criteria(s_logit, t_logit)
        loss_G.backward()
        optimizer_G.step()
    scheduler_G.step()
    return loss_G.item()


def local_update(local_model, global_model, train_dataloader, sample_loader, optimizer, scheduler, epoch_total,
                 alpha, cur_round=0, cuda_available=True):
    """ 客户端本地更新, 包括蒸馏更新和常规更新 """
    criteria_ce = torch.nn.CrossEntropyLoss()
    criteria_l1 = torch.nn.L1Loss()
    criteria_kl = torch.nn.KLDivLoss(reduction="batchmean")

    train_dataiter = iter(train_dataloader)
    sample_iter = iter(sample_loader)

    loss_total = {'loss1': [], 'loss2': []}
    for epoch in range(epoch_total):
        try:
            X, y = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            X, y = next(train_dataiter)
        if cuda_available:
            X, y = X.cuda(), y.cuda()
        local_pred = local_model(X)
        loss_1 = criteria_ce(local_pred, y)

        global_pred = global_model(X)
        loss_2 = criteria_kl(F.log_softmax(local_pred), F.softmax(global_pred))

        try:
            X, y = next(sample_iter)
        except StopIteration:
            sample_iter = iter(sample_loader)
            X, y = next(sample_iter)
        if cuda_available:
            X, y = X.cuda(), y.cuda()
        pred = local_model(X)
        loss_3 = criteria_kl(F.log_softmax(pred, dim=0), F.softmax(y, dim=0))

        if cur_round > 0:
            loss = loss_1 + alpha * loss_2 * alpha * loss_3
        else:
            loss = loss_1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 30)
        optimizer.step()

        loss_total['loss1'].append(loss_1)
        loss_total['loss2'].append(loss_2)
        # print('loss 1 : {}, loss 2 : {}'.format(loss_1.item(), loss_2.item()))

    if scheduler is not None:
        scheduler.step()
    return loss_total


def evaluate(users, model, mode, test_dataloader, criteria):
    """ 测试全局模型或者各个客户端模型的准确率 """
    total_sample = 0
    total_correct = 0
    users_acc = []
    users_loss = []
    for user in users:
        user_correct = 0
        user_num = 0
        user_loss = 0.0
        for X, y in test_dataloader[user]:
            X, y = X.cuda(), y.cuda()
            if mode == 'global':
                pred = model(X)
            else:
                pred = model[user](X)
            loss = criteria(pred, y).item()
            user_loss += loss

            pred = torch.argmax(pred, dim=1)
            user_correct += torch.eq(pred, y).sum().item()
            user_num += y.shape[0]

        total_sample += user_num
        total_correct += user_correct

        user_acc = user_correct / user_num
        users_acc.append(user_acc)
        users_loss.append(loss)
    average_acc = total_correct / total_sample
    return average_acc, users_acc, users_loss


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


def generate_samples(generator, sample_num, nz):
    """ 生成样本， 样本数量是100的倍数 """
    fake_img = []
    for g_idx in range(int(sample_num / 100)):
        z = torch.randn((100, nz, 1, 1)).cuda()
        fake_img_b = generator(z).detach()
        fake_img.extend(fake_img_b[i] for i in range(100))
    fake_img = torch.stack(fake_img)
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
    parser.add_argument('--iid', help='iid or non-iid', type=int, default=0)
    parser.add_argument('--alpha', help='alpha in Dirichelt distribution (smaller means larger heterogeneity)',
                        type=float, default=0.1)
    parser.add_argument("--sampling_ratio", type=float, default=0.1, help="Ratio for sampling training samples.")
    parser.add_argument("--batch_size", type=int, default=32)
    # trainer parameters
    parser.add_argument('--lr_C_local_train', type=float, default=1e-2)
    parser.add_argument('--lr_G_local', type=float, default=1e-4)
    parser.add_argument('--lr_C_global_distill', type=float, default=1)
    parser.add_argument('--lr_C_local_distill', type=float, default=1)
    parser.add_argument('--loss_alpha', type=float, default=1)
    parser.add_argument('--local_epoch', type=int, default=100)
    parser.add_argument('--global_dis_epoch', type=int, default=100)
    # generator parameters
    parser.add_argument('--noise_batch', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--sample_per_round', type=int, default=6400)
    # aux parameters
    parser.add_argument('--distill_loss', type=str, default='kl')
    parser.add_argument('--distill_optim', type=str, default='SGD')
    parsed = parser.parse_args()
    options = parsed.__dict__
    for key, value in options.items():
        print('{}\t:{}'.format(key, value))
    return options


def get_prediction(model, samples):
    dataloader = DataLoader(MiniDataset([samples], [None]), batch_size=128)
    res = []
    for X in dataloader:
        X = X[0].cuda()
        pred = model(X)
        res.extend(pred[i] for i in range(pred.shape[0]))
    res = torch.stack(res)
    return res.detach().cpu()


def main():
    options = read_options()
    if options['alpha'] >= 1:
        options['alpha'] = int(options['alpha'])
    if options['iid'] == 1:
        iid_fold = './data/Mnist/u{}-ratio{}-iid'.format(options['num_user'], options['sampling_ratio'])
        users, train_data, test_data, num_train_data, num_test_data = read_data(iid_fold)
    else:
        non_iid_fold = os.path.join('data', options['dataset'] +
                                    '/u{}c{}-alpha{}-ratio{}'.format(options['num_user'],
                                                                     options['category'],
                                                                     options['alpha'],
                                                                     options['sampling_ratio']))
        users, train_data, test_data, num_train_data, num_test_data = read_data(non_iid_fold)
    suffix = '_'.join(map(str, [arg for arg in options.values()])) + datetime.now(tz_utc_8).strftime("_%m-%d_%H:%M:%S")
    if not os.path.exists('result/' + suffix):
        os.mkdir('result/' + suffix)
        os.mkdir('result/' + suffix + '/log')
        os.mkdir('result/' + suffix + '/checkpoints')
    log_dir = os.path.join('result', suffix, 'log')
    ckpt_dir = os.path.join('result', suffix, 'checkpoints')

    train_dataloader = {user: DataLoader(MiniDataset(list(train_data[user].values()), [None, None]),
                                         batch_size=options['batch_size'], shuffle=True) for user in users}
    test_dataloader = {user: DataLoader(MiniDataset(list(test_data[user].values()), [None, None]),
                                        batch_size=options['batch_size']) for user in users}

    log_writers = {user: SummaryWriter(os.path.join(log_dir, user), flush_secs=5) for user in users}
    log_writers['global'] = SummaryWriter(os.path.join(log_dir, 'global'), flush_secs=5)

    # 全局模型和优化器
    global_model = get_model(options['model'], (1, 28, 28), options['category']).cuda()
    global_generator = get_generator(nz=options['nz'], nc=1, img_size=28).cuda()
    global_optim = optimizers[options['distill_optim']](global_model.parameters(), lr=options['lr_C_global_distill'])
    global_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=global_optim, step_size=1, gamma=1)
    # 客户端模型、生成器、优化器
    user_models = {}
    user_generator = {}
    user_optim_C = {}
    user_optim_dis = {}
    user_optim_G = {}
    user_scheduler_C = {}
    user_scheduler_dis = {}
    user_scheduler_G = {}
    for user in users:
        user_models[user] = get_model(options['model'], (1, 28, 28), options['category']).cuda()
        user_generator[user] = get_generator(nz=options['nz'], nc=1, img_size=28).cuda()

        user_models[user].load_state_dict(copy.deepcopy(global_model.state_dict()))
        user_generator[user].load_state_dict(copy.deepcopy(global_generator.state_dict()))

        user_optim_C[user] = torch.optim.SGD(user_models[user].parameters(), lr=options['lr_C_local_train'],
                                             momentum=0.9, weight_decay=1e-4)
        user_optim_dis[user] = torch.optim.SGD(user_models[user].parameters(), lr=options['lr_C_local_distill'])
        user_optim_G[user] = torch.optim.SGD(user_generator[user].parameters(), lr=options['lr_G_local'])

        user_scheduler_C[user] = torch.optim.lr_scheduler.StepLR(optimizer=user_optim_C[user], step_size=1, gamma=1)
        user_scheduler_dis[user] = torch.optim.lr_scheduler.StepLR(optimizer=user_optim_dis[user],
                                                                   step_size=1, gamma=1)
        user_scheduler_G[user] = torch.optim.lr_scheduler.StepLR(optimizer=user_optim_G[user], step_size=1, gamma=1)
    # 损失计算方法
    criteria_ce = torch.nn.CrossEntropyLoss()
    criteria_l1 = torch.nn.L1Loss()
    criteria_KL = torch.nn.KLDivLoss()

    for user in users:
        train_on_local_data(user_models[user], train_dataloader[user], user_optim_C[user], user_scheduler_C[user],
                            criteria_ce, options['local_epoch'])
    average_acc, users_acc, users_loss = evaluate(users, user_models, 'per', test_dataloader, criteria_ce)
    print('initial users average accuracy {}'.format(average_acc))
    # 联邦训练的过程
    best_acc = 0.0
    # online训练过程
    for round_idx in range(options['comm_round']):
        # 随机挑选客户端 [1]
        selected_users = select_users(users, options['users_per_round'], seed=round_idx)
        # 客户本地更新生成器 [2]
        for user in selected_users:
            train_on_local_data(user_models[user], train_dataloader[user], user_optim_C[user], user_scheduler_C[user],
                                criteria_ce, options['local_epoch'])
            loss_G = train_generator(user_generator[user], user_models[user], global_model, user_optim_G[user],
                                     scheduler_G=user_scheduler_G[user], local_epoch=options['local_epoch'],
                                     batch_size=options['noise_batch'], nz=options['nz'], criteria=criteria_l1)
            log_writers[user].add_scalar('generator training loss', loss_G, round_idx)
        # 聚合生成器 [3]
        param_avg([user_generator[user] for user in selected_users], global_generator)
        for user in selected_users:
            user_generator[user].load_state_dict(copy.deepcopy(global_generator.state_dict()))
        # 生成器产生公共数据 [4]
        samples_gen = generate_samples(global_generator, options['sample_per_round'], options['nz'])
        # 客户端输出预测分数 [5]
        prediction_avg = torch.zeros((samples_gen.shape[0], options['category']))
        for user in selected_users:
            prediction_avg += get_prediction(user_models[user], samples_gen)
        prediction_avg /= len(selected_users)
        # 全局模型的蒸馏过程 [6]
        sample_pred_loader = DataLoader(MiniDataset((samples_gen, prediction_avg.detach()), [None, None]),
                                        batch_size=64)
        loss_global_dis = distill(global_model, sample_pred_loader, global_optim, global_scheduler,
                                  options['global_dis_epoch'], loss_type=options['distill_loss'])
        for loss_idx, loss_item in enumerate(loss_global_dis):
            log_writers['global'].add_scalar('distill loss', loss_item,
                                             loss_idx + round_idx * options['global_dis_epoch'])
        # 全局模型的预测分数 [7]
        if round_idx < 10:
            sample_pred_loader = DataLoader(MiniDataset((samples_gen, prediction_avg.detach()), [None, None]),
                                            batch_size=64, shuffle=True)
        else:
            prediction_global = get_prediction(global_model, samples_gen)
            sample_pred_loader = DataLoader(MiniDataset((samples_gen, prediction_global.detach()), [None, None]),
                                            batch_size=64, shuffle=True)
        # 客户端的本地更新  [8]
        for user in selected_users:
            # loss_local = local_update(user_models[user], global_model, train_dataloader[user], sample_pred_loader,
            #                           user_optim_C[user], user_scheduler_C[user], options['local_epoch'],
            #                           options['loss_alpha'], cur_round=round_idx)
            distill(user_models[user], sample_pred_loader, user_optim_dis[user], user_scheduler_dis[user],
                    options['local_epoch'], loss_type=options['distill_loss'])
            # for idx in range(options['local_epoch']):
            #     train_writer.add_scalar('{} local distillation loss'.format(user), loss_local['loss2'][idx],
            #                             options['local_epoch'] * round_idx + idx)
            #     train_writer.add_scalar('{} local training loss'.format(user), loss_local['loss1'][idx],
            #                             options['local_epoch'] * round_idx + idx)
            pass

        # 测试精度
        average_acc, users_acc, users_loss = evaluate(users, user_models, 'par', test_dataloader, criteria_ce)
        print('round {}, clients average model accuracy {}'.format(round_idx, average_acc))
        for u_idx, user in enumerate(users):
            log_writers[user].add_scalar('acc', users_acc[u_idx], round_idx)
            log_writers[user].add_scalar('loss', users_loss[u_idx], round_idx)

        average_acc, users_acc, users_loss = evaluate(users, global_model, 'global', test_dataloader, criteria_ce)
        print('          global model accuracy {}'.format(average_acc))

        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(global_model, os.path.join(ckpt_dir, 'best_model.pth'))
            torch.save(global_generator, os.path.join(ckpt_dir, 'best_generator.pth'))
        log_writers['global'].add_scalar('average acc', average_acc, round_idx)

    torch.save(global_model, os.path.join(ckpt_dir, 'latest_model.pth'))
    torch.save(global_generator, os.path.join(ckpt_dir, 'latest_generator.pth'))
    print('best global model accuracy {}'.format(best_acc))


if __name__ == "__main__":
    main()
