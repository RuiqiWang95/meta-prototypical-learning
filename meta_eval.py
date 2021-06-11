# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  21:12

import os.path as osp
import argparse

import torch
from torch.utils.data import DataLoader
from meta import MetaLearner
from data_more import MNIST, MiniImageNet, CIFAR, Fashion_MNIST, TaskSampler
from utils import *



def main(opts):
    pprint(vars(opts))
    device = torch.device(opts.device)
    print('using device: {}'.format(device))

    config_file = osp.join('Config', opts.config)
    print('using device: {}'.format(device))
    print('loading meta config from {}'.format(config_file))
    config = torch.load(config_file)
    print(config)

    metalearner = MetaLearner(opts, config)
    metalearner.metanet.load_state_dict(torch.load(opts.weights))
    metalearner.to(device)
    metalearner.eval()

    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        np.random.seed(opts.seed)
        print("\nrandom seed: {}".format(opts.seed))

    if opts.dataset == 'mini':
        test_set = MiniImageNet(mode='test')
    elif opts.dataset == 'MNIST':
        test_set = MNIST(mode='test')
    elif opts.dataset == 'CIFAR':
        test_set = CIFAR(mode='test')
    elif opts.dataset == 'FashionMNIST':
        test_set = Fashion_MNIST(mode='test')
    test_sampler = TaskSampler(test_set.label, 1, opts.n_way,
                               opts.shot_k + opts.query_k)
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler,
                             num_workers=8, pin_memory=True)

    test_accs = []
    for i in range(1000):
        accs_q, loss = metalearner(test_loader, opts.update_step)
        test_accs.append(accs_q)
    test_accs = np.array(test_accs).transpose()

    for i, accs in enumerate(test_accs):
        m, h = mean_confidence_interval(accs)
        print('step {} TEST set  acc: {:.4f}, h: {:.4f}'.format(i, m, h))


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mini')
    parser.add_argument('--task_num', type=int, default=1)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--shot_k', type=int, default=5)
    parser.add_argument('--query_k', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--update_step', type=int, default=10)
    parser.add_argument('--config', type=str, default='metanet.config')
    parser.add_argument('--weights', type=str, default='meta-fiveshot/max-acc.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default='888')
    opts = parser.parse_args()

    main(opts)