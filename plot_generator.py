import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from models import Generator
from untitled import generate_samples


def imshow(imgs):
    img = torchvision.utils.make_grid(imgs)
    width = np.sqrt(len(imgs)) * 1.1
    plt.figure(figsize=(width, width))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


ckpt = torch.load(
    '/home/lichenglong/pycharm_project/dfme/result/fedad_cnn_200_20_10_Mnist_10_0.4_0.25_32_0.01_0.0001_0.0001_0'
    '.0001_20_20_64_100_500_05-25_09:08:58/checkpoints/best_generator.pth')

samples_gen = generate_samples(ckpt, 256, 100)
imshow(samples_gen)
