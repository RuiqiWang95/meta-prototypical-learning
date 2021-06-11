# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  17:00

import argparse

import torch
from torch.utils.data import DataLoader

from data_more import MNIST, TaskSampler
from convnet import Convnet
from pipeline import protoPipe
from utils import *


def main(opts):
    pprint(vars(opts))
    device = torch.device(opts.device)
    print('using device: {}'.format(device))

    encoder = Convnet().to(device)
    encoder.load_state_dict(torch.load(opts.weights))
    encoder.eval()
    pipeline = protoPipe(encoder, opts.shot_k, opts.query_k)

    test_set = MNIST(mode='test')
    test_sampler = TaskSampler(test_set.label, 1000, opts.n_way,
                                opts.shot_k+opts.query_k)
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler,
                             num_workers=8, pin_memory=True)

    test_acc = []
    for episode, batch in enumerate(test_loader, 0):
        task = batch[0].view(opts.n_way * (opts.shot_k + opts.query_k), 3, 84, 84)
        loss, acc = pipeline(task.to(device), opts.n_way)
        test_acc.append(acc)

    m, h = mean_confidence_interval(test_acc)
    print('TEST set  acc: {:.4f}, h: {:.4f}'.format(m, h))


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--shot_k', type=int, default=5)
    parser.add_argument('--query_k', type=int, default=15)
    parser.add_argument('--weights', type=str, default='randomseed-fiveshot/max_acc.pth')
    parser.add_argument('--device', type=str, default='cuda')
    opts = parser.parse_args()

    main(opts)