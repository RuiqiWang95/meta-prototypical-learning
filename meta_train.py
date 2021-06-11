# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  15:47


import os.path as osp
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data import MiniImageNet, MiniImageNetSampler
# from meta_bak import MetaNet, MetaLearner
from meta import MetaLearner
from utils import *


def main(opts):
    pprint(vars(opts))
    ensure_path(opts.name)

    device = torch.device(opts.device)
    config_file = osp.join('Config', opts.config)
    print('using device: {}'.format(device))
    print('loading meta config from {}'.format(config_file))
    config = torch.load(config_file)
    print(config)

    with open(osp.join(opts.name, 'settings.txt'), 'w') as f:
        dic = vars(opts)
        f.writelines(["{}: {}\n".format(k, dic[k]) for k in dic])
        f.writelines(['\nconfig:\n{}'.format(config)])

    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        np.random.seed(opts.seed)
        print("\nrandom seed: {}\n".format(opts.seed))
    writer = SummaryWriter('./{}/run'.format(opts.name))

    train_set = MiniImageNet(mode='train')
    train_sampler = MiniImageNetSampler(train_set.label, opts.task_num, opts.n_way,
                                        opts.shot_k + opts.query_k)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    val_set = MiniImageNet(mode='val')
    val_sampler = MiniImageNetSampler(val_set.label, 1, opts.n_way,
                                      opts.shot_k + opts.query_k)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    # metaencoder = MetaNet(config)
    # print('loading pretrained weights from \'randomseed-fiveshot/max_acc.pth\'')
    # metaencoder.load_pretrained_dict(torch.load('randomseed-fiveshot/max_acc.pth'))
    metalearner = MetaLearner(opts, config)
    metalearner.to(device)
    metalearner.eval()

    optimizer = optim.Adam(metalearner.metanet.parameters(), lr=opts.beta)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    max_val_acc = 0

    for epoch in range(opts.epoch):
        lr_scheduler.step()
        accs_q, loss = metalearner(train_loader, opts.update_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            print('epoch: {}'.format(epoch))
            print('loss: {}'.format(loss.item()))
            # print('shot_losses: {}'.format(losses_s))
            print('query_accs: {}'.format(accs_q))

            writer.add_scalar("train_loss", loss.item(), epoch)
            # for i in range(len(losses_s)):
            #     writer.add_scalar("shot_loss_{}".format(i), losses_s[i], epoch)

            for i in range(len(accs_q)):
                writer.add_scalar('query_acc_{}'.format(i), accs_q[i], epoch)

        if epoch%100 == 0:
            metalearner.eval()
            val_accs = []
            for i in range(100):
                accs_q, loss = metalearner(val_loader, opts.test_update_step)
                val_accs.append(accs_q[-1])
            m, h = mean_confidence_interval(val_accs)
            writer.add_scalar("val_acc", m, epoch)
            print('VAL set  acc: {:.4f}, h: {:.4f}'.format(m, h))
            if epoch % 1000 == 0:
                torch.save(metalearner.metanet.state_dict(), osp.join(opts.name, 'epoch_{}.pth'.format(epoch)))
            if m > max_val_acc:
                max_val_acc = m
                torch.save(metalearner.metanet.state_dict(), osp.join(opts.name, 'max_acc.pth'))
            metalearner.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30000)
    parser.add_argument('--config', type=str, default='metanet_5.config')
    parser.add_argument('--task_num', type=int, default=1)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--shot_k', type=int, default=5)
    parser.add_argument('--query_k', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--update_step', type=int, default=5)
    parser.add_argument('--test_update_step', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default='meta_test')
    parser.add_argument('--seed', type=int, default=None)
    opts = parser.parse_args()

    main(opts)