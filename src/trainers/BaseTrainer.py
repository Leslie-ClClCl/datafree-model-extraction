import torch
import numpy as np
from ..models.worker import Worker
from ..models.client import Client
import time

from ..utils.worker_utils import Metrics


class BaseTrainer(object):
    def __init__(self, dataset, options, model=None, optimizer=None, worker=None):
        # 为训练器设置一个worker
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')

        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))
        # 为训练器设置参数
        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_avg = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_avg else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([options['name'], f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()

    @staticmethod
    def move_model_to_gpu(model):
        model.cuda()

    def setup_clients(self, dataset):
        """
        实例化客户端
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """
        训练器的训练过程
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        """
        按照设置的每轮参与训练的用户数量，随机挑选客户端
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients):
        """
        选中的客户端的本地更新过程
        """
        solutions = []
        for idx, client in enumerate(selected_clients):
            # 载入最新的全局模型
            client.set_flat_model_params(self.latest_model)
            # 客户端本地更新
            solution = client.local_train()
            solutions.append(solution)
        return solutions

    def aggregate(self, solutions):
        """
        聚合本地客户端的参数，输出一个全局的参数
        """
        avg_solution = torch.zeros_like(self.latest_model)
        # 两种聚合方式：1）简单的平均聚合 2）根据数据量进行加权聚合
        if self.simple_avg:
            num = 0
            for num_sample, local_solution in solutions:
                avg_solution += local_solution
                num += 1
            avg_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solutions:
                avg_solution += num_sample * local_solution
                num += num_sample
            avg_solution /= num
        return avg_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # Measure the gradient difference
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                stats_from_train_data['gradnorm'], difference, end_time - begin_time))
            print('=' * 102 + "\n")
        return global_grads

    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                round_i, stats_from_eval_data['acc'],
                stats_from_eval_data['loss'], end_time - begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}
        return stats
