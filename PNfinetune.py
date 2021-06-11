# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  17:17

import argparse
import os.path as osp
import numpy as np

import torch
from torch.utils.data import DataLoader

from data_more import MNIST, MiniImageNet, CIFAR, TaskSampler
from meta import ProtoMetaLearner
from utils import *


def main(opts):
    pprint(vars(opts))
    device = torch.device(opts.device)
    print('using device: {}'.format(device))

    device = torch.device(opts.device)
    config_file = osp.join('Config', opts.config)
    print('using device: {}'.format(device))
    print('loading meta config from {}'.format(config_file))
    config = torch.load(config_file)
    # print(config)
    metalearner = ProtoMetaLearner(opts, config)



    metalearner.metanet.loadfromPM(opts.weights)
    metalearner.to(device)
    metalearner.eval()

    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        np.random.seed(opts.seed)
        print("\nrandom seed: {}".format(opts.seed))

    if opts.dataset == 'mini':
        test_set = MiniImageNet(mode='test')
    elif opts.dataset=='MNIST':
        test_set = MNIST(mode='test')
    elif opts.dataset=='CIFAR':
        test_set = CIFAR(mode='test')
    test_sampler = TaskSampler(test_set.label, 1, opts.n_way,
                               opts.shot_k+opts.query_k)
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler,
                             num_workers=8, pin_memory=True)

    test_accs = []
    for i in range(1000):
        accs_q, loss = metalearner(test_loader, opts.test_update_step, opts.n_way, opts.shot_k)
        test_accs.append(accs_q)
    test_accs = np.array(test_accs).transpose()

    for i, accs in enumerate(test_accs):
        m, h = mean_confidence_interval(accs)
        print('step {} TEST set  acc: {:.4f}, h: {:.4f}'.format(i, m, h))


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mini')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--shot_k', type=int, default=5)
    parser.add_argument('--query_k', type=int, default=15)
    parser.add_argument('--config', type=str, default='protometanet.config')
    parser.add_argument('--weights', type=str, default='prototypical-networks/max_acc.pth')
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--test_update_step', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=888)
    opts = parser.parse_args()

    main(opts)