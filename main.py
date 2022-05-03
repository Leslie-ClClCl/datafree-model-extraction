import numpy as np
import argparse
import importlib
import torch
import os

from src.trainers.fedad import FedAdTrainer
from src.utils.worker_utils import read_data
from src.trainers.fedavg import FedAvgTrainer
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('rm tmp')


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedad')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_1_random_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=5e-4)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=1e-5)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parser.add_argument('--name',
                        help='description of this experiments',
                        type=str,
                        default='Test')
    parser.add_argument('--local_epoch',
                        help='# local distillation epoch',
                        type=int,
                        default=20)
    parser.add_argument('--noise_batch',
                        type=int,
                        default=64)
    parser.add_argument('--lr_C_local_train',
                        type=float,
                        default=1e-2)
    parser.add_argument('--lr_G_local',
                        type=float,
                        default=1e-4)
    parser.add_argument('--lr_C_global_distill',
                        type=float,
                        default=1e-4)
    parser.add_argument('--lr_C_local_distill',
                        type=float,
                        default=1e-4)
    parser.add_argument('--distill_loss',
                        type=str,
                        default='L1Loss')
    parser.add_argument('--nz', type=int, default=100)
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_class = FedAdTrainer

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
