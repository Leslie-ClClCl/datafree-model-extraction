import time

import numpy as np
import torch
from torch.utils.data import DataLoader


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, train_data, test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker

        self.train_data = train_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """Get model gradient"""
        grad_in_tenser = self.worker.get_flat_grads(self.train_dataloader)
        return grad_in_tenser.cpu().detach().numpy()

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        grads = self.get_flat_grads()  # Return grad in numpy array

        return (len(self.train_data), grads), stats

    def local_train(self, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, **kwargs)
        end_time = time.time()

        stats = {'id': self.cid, "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return len(self.train_data), local_solution

    def local_test(self, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.worker.local_test(dataloader)

        return tot_correct, len(dataset), loss


class AdClient(Client):
    def __init__(self, cid, group, train_data, test_data, batch_size, worker):
        self.local_train_round_ctr = 0
        super(AdClient, self).__init__(cid, group, train_data, test_data, batch_size, worker)

    def local_train(self, latest_model=None, train_G=True):
        # 首先客户端在本地训练数据集上进行训练
        self.local_train_round_ctr += self.worker.train_on_private_data(self.cid, self.train_dataloader,
                                                                        self.local_train_round_ctr)
        if train_G:
            solution = self.worker.local_train_G(self.cid, latest_model)
            return len(self.train_data), solution
        return len(self.train_data)

    def local_test(self, round_i):
        self.worker.test_local_model(self.cid, self.test_dataloader, round_i)

    def set_latest_G(self, latest_G):
        self.worker.set_latest_G(self.cid, latest_G)

    def pred(self, input_loader):
        return self.worker.pred(self.cid, input_loader)

    def local_distill(self, samples, global_pred, round_i):
        self.worker.local_distill(self.cid, samples, global_pred, round_i)

    def test_global_model(self, latest_model, criteria):
        total_samples = 0
        correct = 0
        loss = 0
        for idx, (X, y) in enumerate(self.test_dataloader):
            X, y = X.cuda(), y.cuda()
            pred = latest_model(X)
            loss += criteria(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            correct += torch.eq(pred, y).sum().item()
            total_samples += y.shape[0]
        return correct, loss, total_samples

