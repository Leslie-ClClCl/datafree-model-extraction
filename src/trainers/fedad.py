import copy
import os
import time

import torch
from tensorboardX import SummaryWriter
from torch import optim

from ..trainers.BaseTrainer import BaseTrainer
from ..models.ad_worker import AdWorker
from ..models.client import AdClient
from ..models.model import choose_model, Generator
from ..utils.worker_utils import MiniDataset, set_flat_model_params, get_flat_model_params, mkdir
from torch.utils.data import DataLoader


class FedAdTrainer(BaseTrainer):
    def __init__(self, options, dataset):
        # 设置全局模型
        self.latest_G = Generator(nz=options['nz'], nc=1, img_size=28).cuda()
        self.latest_model = choose_model(options).cuda()
        # 为每个客户端添加分类器和生成器
        models = []
        generators = []
        optimizer_C = []
        optimizer_G = []
        for idx in range(len(dataset[0])):
            C = choose_model(options).cuda()
            G = Generator(nz=options['nz'], nc=1, img_size=28).cuda()  # TODO 为生成器添加合适的参数 MNIST
            C.load_state_dict(copy.deepcopy(self.latest_model.state_dict()))
            G.load_state_dict(copy.deepcopy(self.latest_G.state_dict()))
            models.append(C)
            generators.append(G)
            optimizer_C.append(optim.SGD(C.parameters(), lr=options['lr'], weight_decay=options['wd'], momentum=0.9))
            optimizer_G.append(optim.Adam(G.parameters(), lr=options['lr_G']))
        # 日志保存路径
        result_path = mkdir(os.path.join('./result', options['dataset']))
        suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format('_'.join([options['name'],
                                                              f'tn{len(models)}']),
                                                    options['seed'],
                                                    options['lr'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    'w' if options['noaverage'] else 'a')
        exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                        options['model'], suffix)
        self.eval_event_fold = mkdir(os.path.join(result_path, exp_name, 'eval.event'))
        self.train_event_fold = mkdir(os.path.join(result_path, exp_name, 'train.event'))

        # 设置一个worker
        worker = AdWorker(models, generators, [optimizer_C, optimizer_G],
                          [self.train_event_fold, self.eval_event_fold], options)
        #
        self.local_epoch = options['local_epoch']
        self.nz = options['nz']
        self.batch_size = options['batch_size']
        super(FedAdTrainer, self).__init__(dataset, options, worker=worker, client_class=AdClient)

    def local_pred(self, round_i, selected_client, fake_img):
        predictions = []
        for idx, client in enumerate(selected_client):
            # 载入最新的全局生成器
            flatten_G = get_flat_model_params(self.latest_G)
            client.set_latest_G(flatten_G)
            # 客户端预测得到结果
            pred = client.pred(fake_img).detach()
            predictions.append(pred)
        return predictions

    def global_distill(self, pred_from_clients, fake_img):
        # 对客户端的预测分数进行平均
        pred_from_clients_mean = torch.mean(torch.stack(pred_from_clients), dim=0)
        dataloader = DataLoader(MiniDataset(fake_img.cpu(), pred_from_clients_mean.cpu()), batch_size=64)
        # TODO 选择服务端蒸馏的优化器和损失函数参数
        optimizer = torch.optim.SGD(self.latest_model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
        criteria = torch.nn.L1Loss()
        writer = SummaryWriter(self.train_event_fold, flush_secs=5)
        self.latest_model.train()
        for round_i in range(self.num_round):
            distill_loss = 0
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                X, y = X.cuda(), y.cuda()
                global_pred = self.latest_model(X)
                loss = criteria(global_pred, y)
                loss.backward()
                distill_loss += loss.item()
                optimizer.step()
            writer.add_scalar('global model distillation loss', distill_loss, round_i)
        # 返回全局模型的预测结果
        writer.close()
        latest_pred = self.latest_model(fake_img)
        return latest_pred.detach()

    def test_latest_model_on_traindata(self, round_i):
        total = 0
        correct = 0
        loss = 0
        for client in self.clients:
            correct_c, loss_c, total_c = client.test_global_model(self.latest_model, torch.nn.CrossEntropyLoss())
            total += total_c
            correct += correct_c
            loss += loss_c
        print('acc {}'.format(correct/total))
        self.metrics.update_eval_stats(round_i, {'loss': loss / total, 'acc': correct / total})

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        for round_i in range(self.num_round):
            print('\t>>>round {}'.format(round_i))
            # TODO 每个round开始时测试精度
            # 随机选择客户端
            selected_clients = self.select_clients(seed=round_i)
            # 客户端在本地更新生成器G
            solutions_G = []
            for idx, client in enumerate(selected_clients):
                solution_G = client.local_train(self.latest_model)
                solutions_G.append(solution_G)
            # 聚合生成器
            set_flat_model_params(self.latest_G, self.aggregate(solutions_G))
            # 规定这一轮的统一噪声
            z = torch.randn((self.nz, self.batch_size, 1, 1)).cuda()
            fake_img = self.latest_G(z).detach()
            # 客户端在G生成的样本上输出预测分数
            pred_G = self.local_pred(round_i, selected_clients, fake_img)
            # 服务器的模型的蒸馏，并输出服务端模型的预测分数
            pred_latest = self.global_distill(pred_G, fake_img)
            # 客户端的本地蒸馏
            for idx, client in enumerate(selected_clients):
                client.local_distill(fake_img, pred_latest)

            # 测试精度
            self.test_latest_model_on_traindata(round_i)  # global model的准确率
            for client in self.clients:  # 对每个客户端测试准确率
                client.local_test(round_i)
        # Save tracked information
        self.metrics.write()
