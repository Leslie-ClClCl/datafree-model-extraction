from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from generate_niid_dirichlet import get_dataset
random.seed(42)
np.random.seed(42)


def get_whole_dataset(mode='train'):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST(root='./data', train=True if mode == 'train' else False, download=True, transform=transform)
    n_sample = len(dataset.data)
    SRC_N_CLASS = len(dataset.classes)
    # full batch
    trainloader = DataLoader(dataset, batch_size=n_sample, shuffle=False)

    print("Loading data from storage ...")
    for _, xy in enumerate(trainloader, 0):
        dataset.data, dataset.targets = xy

    index = list(range(dataset.targets.shape[0]))
    random.shuffle(index)
    dataset.data, dataset.targets = dataset.data[index], dataset.targets[index]

    return dataset.data, dataset.targets, len(index)


def process_data(mode, n_user, data, target):

    dataset = {'users': [], 'user_data': {}, 'num_samples': []}
    saving_path = f'./{path_prefix}/{mode}/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    saving_path += f'{mode}.pt'
    for user_idx in range(n_user):
        user_data = data[user_idx * sample_per_user: (user_idx + 1) * sample_per_user]
        user_target = target[user_idx * sample_per_user: (user_idx + 1) * sample_per_user]
        uname = 'iid_{0:05d}'.format(user_idx)
        dataset['users'].append(uname)
        dataset['user_data'][uname] = {
            'x': torch.tensor(user_data, dtype=torch.float32),
            'y': torch.tensor(user_target, dtype=torch.int64)}
        dataset['num_samples'].append(sample_per_user)
    with open(saving_path, 'wb') as outfile:
        print(f"Dumping data => {saving_path}")
        torch.save(dataset, outfile)


if __name__ == '__main__':
    user_num = 20
    sample_ratio = 0.25
    path_prefix = f'./data/Mnist/u{user_num}-ratio{sample_ratio}-iid'
    train_X, train_y, train_sample_num = get_whole_dataset('train')
    sample_per_user = int((train_sample_num * sample_ratio / user_num))
    process_data('train', user_num, train_X, train_y)

    test_X, test_y, test_sample_num = get_whole_dataset('test')
    sample_per_user = int((test_sample_num / user_num))
    process_data('test', user_num, test_X, test_y)
