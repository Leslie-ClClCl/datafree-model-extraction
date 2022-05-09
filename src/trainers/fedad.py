import copy
import os
import time
import threading
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim

from ..trainers.BaseTrainer import BaseTrainer
from ..models.ad_worker import AdWorker
from ..models.client import AdClient
from ..models.model import choose_model, Generator
from ..utils.worker_utils import MiniDataset, set_flat_model_params, get_flat_model_params, mkdir
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
            G = Generator(nz=options['nz'], nc=1, img_size=28).cuda()
            C.load_state_dict(copy.deepcopy(self.latest_model.state_dict()))
            G.load_state_dict(copy.deepcopy(self.latest_G.state_dict()))
            models.append(C)
            generators.append(G)
            optimizer_C.append(optim.SGD(C.parameters(), lr=options['lr_C_local_distill'],
                                         weight_decay=options['wd'], momentum=0.9))
            optimizer_G.append(optim.RMSprop(G.parameters(), lr=options['lr_G_local']))
        # 日志保存路径
        result_path = mkdir(os.path.join('./result', options['dataset']))
        suffix = '{}_sd{}_lr0{}_lr1{}_lr2{}_lr3{}_ep{}_bs{}_{}'.format('_'.join([options['name'],
                                                              f'tn{len(models)}']),
                                                    options['seed'],
                                                    options['lr_C_local_train'],
                                                                       options['lr_G_local'],
                                                                       options['lr_C_global_distill'],
                                                                       options['lr_C_local_distill'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    'w' if options['noaverage'] else 'a')
        exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                        options['model'], suffix)
        eval_event_fold = mkdir(os.path.join(result_path, exp_name, 'eval.event'))
        train_event_fold = mkdir(os.path.join(result_path, exp_name, 'train.event'))
        self.train_writer = SummaryWriter(train_event_fold, flush_secs=5)
        self.eval_writer = SummaryWriter(eval_event_fold, flush_secs=5)

        # 设置一个worker
        worker = AdWorker(models, generators, [optimizer_C, optimizer_G],
                          [self.train_writer, self.eval_writer], options)
        #
        self.local_epoch = options['local_epoch']
        self.nz = options['nz']
        self.batch_size = options['batch_size']
        self.lr_C_global_distill = options['lr_C_global_distill']
        super(FedAdTrainer, self).__init__(dataset, options, worker=worker, client_class=AdClient)

    def local_pred(self, round_i, selected_client, fake_img_loader):
        predictions = []
        for idx, client in enumerate(selected_client):
            # 载入最新的全局生成器
            flatten_G = get_flat_model_params(self.latest_G)
            client.set_latest_G(flatten_G)
            # 客户端预测得到结果
            pred = client.pred(fake_img_loader).detach()
            predictions.append(pred)
        return predictions

    def global_distill(self, pred_from_clients, fake_img, round_i):
        # 对客户端的预测分数进行平均
        pred_from_clients_mean = torch.mean(torch.stack(pred_from_clients), dim=0).cpu()
        pred_from_clients_soft = F.softmax(pred_from_clients_mean, dim=-1)
        dataloader = DataLoader(MiniDataset(fake_img.cpu(), pred_from_clients_soft, label_type='float32'),
                                batch_size=64)
        # TODO 选择服务端蒸馏的优化器和损失函数参数
        optimizer = torch.optim.RMSprop(self.latest_model.parameters(), lr=self.lr_C_global_distill)
        # criteria = torch.nn.L1Loss()
        criteria = torch.nn.KLDivLoss(reduction="batchmean")
        writer = self.train_writer
        self.latest_model.train()
        for r_idx in range(self.num_round):
            distill_loss = 0
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                X, y = X.cuda(), y.cuda()
                global_pred = self.latest_model(X)
                global_pred = F.log_softmax(global_pred, dim=-1)
                loss = criteria(global_pred, y)
                loss.backward()
                optimizer.step()
                distill_loss += loss.item()
            writer.add_scalar('global model distillation loss', distill_loss, round_i * self.num_round + r_idx)
        # 返回全局模型的预测结果
        latest_pred = []
        for X, _ in dataloader:
            X = X.cuda()
            pred = self.latest_model(X).detach().cpu()
            latest_pred.extend([pred[i] for i in range(X.shape[0])])
        latest_pred = torch.stack(latest_pred)
        return latest_pred

    def test_latest_model_on_traindata(self, round_i):
        total = 0
        correct = 0
        loss = 0
        for client in self.clients:
            correct_c, loss_c, total_c = client.test_global_model(self.latest_model, torch.nn.CrossEntropyLoss())
            total += total_c
            correct += correct_c
            loss += loss_c
        accuracy = correct/total
        print('acc {}'.format(accuracy))
        self.eval_writer.add_scalar('global model accuracy', accuracy, round_i)
        return accuracy

    def train(self):
        # pretrain
        c_threads = []
        for client in self.clients:
            client_thread = threading.Thread(target=client.local_train, args=(None, False))
            client_thread.start()
            time.sleep(0.5)
            c_threads.append(client_thread)
        for c_thread in c_threads:
            c_thread.join()
        fake_img = []
        for g_idx in range(100):
            z = torch.randn((100, 100, 1, 1)).cuda()
            fake_img_b = self.latest_G(z).detach().cpu()
            fake_img.extend(fake_img_b[i] for i in range(100))
        fake_img = torch.stack(fake_img)
        fake_img_loader = DataLoader(MiniDataset(fake_img.cpu(), np.array([0] * fake_img.shape[0])),
                                     shuffle=False, batch_size=64)
        # 客户端在G生成的样本上输出预测分数
        pred_clients = self.local_pred(0, self.clients, fake_img_loader)
        # 服务器的模型的蒸馏，并输出服务端模型的预测分数
        self.global_distill(pred_clients, fake_img, 0)

        best_acc = 0.0
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        for round_i in range(self.num_round):
            print('\t>>>round {}'.format(round_i))
            # 随机选择客户端
            selected_clients = self.select_clients(seed=round_i)
            # 客户端在本地更新生成器G
            solutions_G = []
            c_threads = []
            for idx, client in enumerate(selected_clients):
                c_threads.append(threading.Thread(target=client.local_train, args=self.latest_model))
                solution_G = client.local_train(self.latest_model)
                solutions_G.append(solution_G)
            # 聚合生成器
            set_flat_model_params(self.latest_G, self.aggregate(solutions_G))
            # 生成器通过噪声产生公共数据
            fake_img = []
            for g_idx in range(100):
                z = torch.randn((100, 100, 1, 1)).cuda()
                fake_img_b = self.latest_G(z).detach().cpu()
                fake_img.extend(fake_img_b[i] for i in range(100))
            fake_img = torch.stack(fake_img)

            fake_img_loader = DataLoader(MiniDataset(fake_img.cpu(), np.array([0] * fake_img.shape[0])),
                                         shuffle=False, batch_size=64)
            # 客户端在G生成的样本上输出预测分数
            pred_G = self.local_pred(round_i, selected_clients, fake_img_loader)
            # 服务器的模型的蒸馏，并输出服务端模型的预测分数
            pred_latest = self.global_distill(pred_G, fake_img, round_i)
            # 客户端的本地蒸馏
            for idx, client in enumerate(selected_clients):
                client.local_distill(fake_img, pred_latest, round_i)

            # 测试精度
            global_model_acc = self.test_latest_model_on_traindata(round_i)  # global model的准确率
            if global_model_acc > best_acc:
                best_acc = global_model_acc
                torch.save(self.latest_model, 'checkpoints/global_model/best_global_model.pt')
                torch.save(self.latest_G, 'checkpoints/global_model/best_global_generator.pt')
            for client in self.clients:  # 对每个客户端测试准确率
                client.local_test(round_i)
        print('>>>> FedAD Train Done!')
