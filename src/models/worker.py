import numpy as np
import torch
import torch.nn as nn


class Worker(object):
    """
    联邦学习算法中的worker
    """
    def __init__(self, model, optimizer, options):
        # 初始化参数
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = options['num_epoch']

    def get_model_param(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        """
        从文件读取模型参数，并加载到models
        """
        model_params_dict = torch.load(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flatten_params = []
        for params in self.model.parameters():
            flatten_params.append(params.data.view(-1))
        flatten_params = torch.cat(flatten_params)
        return flatten_params.detach()

    def set_flat_model_params(self, flatten_params):
        prev_idx = 0
        for params in self.model.parameters():
            flat_size = int(np.prod(list(params.size())))
            params.data.copy_(
                flatten_params[prev_idx:prev_idx + flat_size].view(params.size()))
            prev_idx += flat_size

    def get_flat_grads(self, dataloader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flatten_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flatten_grads

    def local_train(self, train_dataloader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.model.train()
        train_loss = train_acc = train_total = 0.0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(), "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        return_dict = {"loss": train_loss / train_total, "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss
