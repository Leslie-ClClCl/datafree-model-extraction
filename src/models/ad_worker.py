from torch.utils.data import DataLoader

from src.utils.worker_utils import MiniDataset, get_flat_model_params, set_flat_model_params, mkdir
from ..models.worker import Worker
import torch
import torch.nn.functional as F

from ..optimizer.gd import GD


class AdWorker(Worker):
    def __init__(self, models, generators, optimizers, event_writer, options):
        self.models = models
        self.generators = generators
        self.optim_C, self.optim_G = optimizers

        self.local_epoch = options['local_epoch']
        self.nz = options['nz']
        self.batch_size = options['batch_size']
        self.options = options
        self.train_writer, self.eval_writer = event_writer

    def train_on_private_data(self, client_id, train_loader, round_i):
        """
        客户端在本地训练数据上训练
        """
        criteria = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.models[client_id].parameters(), lr=self.options['lr_C_local_train'])
        # optimizer = GD(self.models[client_id].parameters(), lr=self.options['lr'], weight_decay=self.options['wd'])
        optimizer = torch.optim.SGD(self.models[client_id].parameters(), lr=self.options['lr_C_local_train'],
                                    weight_decay=self.options['wd'], momentum=0.9)
        if round_i > 25:
            optimizer = torch.optim.SGD(self.models[client_id].parameters(), lr=self.options['lr_C_local_train']/10,
                                        weight_decay=self.options['wd'], momentum=0.9)
        for i in range(self.local_epoch):
            train_loss = 0
            for idx, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                X, y = X.cuda(), y.cuda()
                pred = self.models[client_id](X)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criteria(pred, y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.models[client_id].parameters(), 60)
                optimizer.step()

                train_loss += loss.item()
            self.train_writer.add_scalar('client {} training loss'.format(client_id), train_loss, i+round_i)
        return self.local_epoch

    def test_local_model(self, client_id, test_loader, round_i):
        total = 0
        correct = 0
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            pred = self.models[client_id](X)
            pred = torch.argmax(pred, dim=1)
            correct += torch.eq(pred, y).sum().item()
            total += y.shape[0]
        self.eval_writer.add_scalar('client {} test accuracy'.format(client_id), correct/total, round_i)

    def local_train_G(self, client_id, latest_model):
        """
        客户端的本地更新，自己本地模型与服务端模型，训练生成器G，
        生成令这两个模型输出差异最大的、以及对当前服务器模型更新最有益的样本
        """
        optimizer_C = self.optim_C[client_id]
        optimizer_G = self.optim_G[client_id]
        for i in range(self.local_epoch):
            for k in range(5):
                z = torch.randn((self.batch_size, self.nz, 1, 1)).cuda()
                optimizer_C.zero_grad()
                fake = self.generators[client_id](z).detach()
                t_logit = latest_model(fake)
                s_logit = self.models[client_id](fake)
                loss_C = F.l1_loss(s_logit, t_logit.detach())

                loss_C.backward()
                optimizer_C.step()

            z = torch.randn((self.batch_size, self.nz, 1, 1)).cuda()
            optimizer_G.zero_grad()
            self.generators[client_id].train()
            fake = self.generators[client_id](z)
            t_logit = latest_model(fake)
            s_logit = self.models[client_id](fake)

            loss_G = - F.l1_loss(s_logit, t_logit)

            loss_G.backward()
            optimizer_G.step()
        return get_flat_model_params(self.generators[client_id])

    def set_latest_G(self, client_id, latest_G):
        set_flat_model_params(self.generators[client_id], latest_G)

    def pred(self, client_id, sample_loader):
        res = []
        for X, _ in sample_loader:
            X = X.cuda()
            pred = self.models[client_id](X)
            res.extend([pred[i] for i in range(pred.shape[0])])
        res = torch.stack(res)
        return res

    def local_distill(self, client_id, samples, global_pred, round_i):
        sample_num = samples.shape[0]
        # global_pred = F.softmax(global_pred, dim=-1)
        dataloader = DataLoader(MiniDataset(samples, global_pred, label_type='float32'), shuffle=False, batch_size=64)
        optimizer = torch.optim.Adam(self.models[client_id].parameters(), lr=self.options['lr_C_local_distill'])
        criteria = torch.nn.L1Loss()
        # criteria = torch.nn.KLDivLoss(reduction="batchmean")
        for epoch_idx in range(self.local_epoch):
            local_distill_loss = 0
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                X, y = X.cuda(), y.cuda()
                local_pred = self.models[client_id](X)
                # local_pred = F.log_softmax(local_pred, dim=-1)
                loss = criteria(local_pred, y)
                loss.backward()
                optimizer.step()

                local_distill_loss += loss.item()
            self.train_writer.add_scalar('client {} local distillation loss'.format(client_id),
                                         local_distill_loss, round_i*self.local_epoch+epoch_idx)
        print('client {} local distillation done'.format(client_id))
