from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data


class ShadowCIFAR10(CIFAR10):

    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        if self.train:
            if target:
                # target train size 0:config.general.train_target_size
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:
                # shadow train size config.general.train_target_size:-1
                self.data = self.data[config.general.train_target_size:]
                self.targets = self.targets[
                               config.general.train_target_size * (num + 1):config.general.train_target_size * (
                                       num + 2)]
        else:
            if target:
                # target test size 0:config.general.test_target_size
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]
            else:
                # shadow test size config.general.test_target_size:-1
                self.data = self.data[config.general.test_target_size:]
                self.targets = self.targets[
                               config.general.test_target_size * (num + 1):config.general.test_target_size * (num + 2)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class custum_MNIST(MNIST):

    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        if self.train:
            if target:
                # target train size 0:config.general.train_target_size
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:
                # shadow train size config.general.train_target_size:-1
                self.data = self.data[config.general.train_target_size:]
                self.targets = self.targets[
                               config.general.train_target_size * (num + 1):config.general.train_target_size * (
                                       num + 2)]
        else:
            if target:
                # target test size 0:config.general.test_target_size
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]
            else:
                # shadow test size config.general.test_target_size:-1
                self.data = self.data[config.general.test_target_size:]
                self.targets = self.targets[
                               config.general.test_target_size * (num + 1):config.general.test_target_size * (num + 2)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_data_for_final_eval(models, all_dataloaders, device):
    Y = []
    X = []
    C = []
    for idx_model, model in enumerate(models):
        model.eval()
        # print(all_dataloaders)
        dataloaders = all_dataloaders[idx_model]
        for phase in ['train', 'val']:
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)
                output = model(inputs)
                for out in output.cpu().detach().numpy():
                    X.append(out)
                    if phase == "train":
                        Y.append(1)
                    else:
                        Y.append(0)
                for cla in labels.cpu().detach().numpy():
                    C.append(cla)
    return (np.array(X), np.array(Y), np.array(C))


transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


class CustomCIFAR(Dataset):
    def __init__(self, data, target, transform=None, raw_class=None):
        self.data = data
        self.target = target
        self.raw_class = raw_class
        self.tranform = transform

    def __getitem__(self, index):
        x = self.data[index]
        x = Image.fromarray(x)
        if self.tranform is not None:
            x = self.tranform(x)
        y = self.target[index]
        if self.raw_class is not None:
            c = self.raw_class[index]
            return x, y, c
        return x, y

    def __len__(self):
        return len(self.target)


def get_average_cifar10():
    # 将训练数据集和测试数据集全部划分为3W张
    cifar_train = datasets.CIFAR10(root='../data', train=True)
    cifar_test = datasets.CIFAR10(root='../data', train=False)
    cifar_train_avg = CustomCIFAR(cifar_train.data[0:30000], target=cifar_train.targets[0:30000],
                                  transform=transform_cifar)
    cifar_test_avg = CustomCIFAR(data=np.concatenate((cifar_test.data, cifar_train.data[30000:50000])),
                                 target=np.concatenate((cifar_test.targets, cifar_train.targets[30000:50000])),
                                 transform=transform_cifar)
    return cifar_train_avg, cifar_test_avg


def split_cifar_for_attack(train_set:CustomCIFAR, test_set:CustomCIFAR):
    # 将Target Model的训练集和测试集划分成两部分
    # 分别是Attack Model的训练集和测试集，前者2.5W+2.5W=5W, 后者0.5W+0.5W=1W张
    target_train = np.array([1] * 25000 + [0] * 25000)
    target_test = np.array([1] * 5000 + [0] * 5000)
    attack_train = CustomCIFAR(data=np.concatenate((train_set.data[0:25000], test_set.data[0:25000])), target=target_train,
                               transform=transform_cifar)
    attack_test = CustomCIFAR(data=np.concatenate((train_set.data[25000:30000], test_set.data[25000:30000])), target=target_test,
                              transform=transform_cifar)
    return attack_train, attack_test


if __name__ == '__main__':
    cifar_train_avg, cifar_test_avg = get_average_cifar10()
    com = split_cifar_for_attack(cifar_train_avg, cifar_test_avg)
    pass
