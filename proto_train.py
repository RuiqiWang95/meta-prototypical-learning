# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  15:27

import os.path as osp
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data import MiniImageNet, MiniImageNetSampler
from convnet import Convnet
from pipeline import protoPipe
from utils import *


def main(opts):
    pprint(vars(opts))
    ensure_path(opts.name)
    with open(osp.join(opts.name, 'settings.txt'), 'w') as f:
        dic = vars(opts)
        f.writelines(["{}: {}\n".format(k, dic[k]) for k in dic])

    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        np.random.seed(opts.seed)
        print("\nrandom seed: {}".format(opts.seed))
    writer = SummaryWriter('./{}/run'.format(opts.name))

    device = torch.device(opts.device)
    print('using device: {}'.format(device))

    train_set = MiniImageNet(mode='train')
    train_sampler = MiniImageNetSampler(train_set.label, 100, opts.c_sampled,
                                        opts.shot_k+opts.query_k)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    val_set = MiniImageNet(mode='val')
    val_sampler = MiniImageNetSampler(val_set.label, 400, opts.n_way,
                                      opts.shot_k+opts.query_k)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    encoder = Convnet().to(device)
    # print(encoder.state_dict()['block4.0.bias'])
    pipeline = protoPipe(encoder, opts.shot_k, opts.query_k)
    pipeline.train()

    optimizer = optim.Adam(encoder.parameters(), opts.lr, )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    max_val_acc = 0
    for epoch in range(opts.epoch // 100):

        for episode, batch in enumerate(train_loader, 0):
            lr_scheduler.step()

            task = batch[0].view(opts.c_sampled * (opts.shot_k + opts.query_k), 3, 84, 84)
            loss, acc = pipeline(task.to(device), opts.c_sampled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if episode % 10 == 0:
                writer.add_scalar("train_loss", loss.item(), epoch*100+episode)
                print('epoch: {}, episode: {} ,loss: {:.4f},acc: {:.4f}'.format(epoch, episode, loss.item(), acc))

        val_acc = []
        pipeline.eval()
        for episode, batch in enumerate(val_loader, 0):
            task = batch[0].view(opts.n_way * (opts.shot_k + opts.query_k), 3, 84, 84)
            loss, acc = pipeline(task.to(device), opts.n_way)
            val_acc.append(acc)

        m, h = mean_confidence_interval(val_acc)
        writer.add_scalar("val_acc", m, epoch*100)
        print('VAL set  acc: {:.4f}, h: {:.4f}'.format(m, h))
        if epoch % (opts.epoch//20) == 0:
            torch.save(pipeline.encoder.state_dict(), osp.join(opts.name, 'epoch_{}.pth'.format(epoch)))
        if m > max_val_acc:
            max_val_acc = m
            torch.save(pipeline.encoder.state_dict(), osp.join(opts.name, 'max_acc.pth'))
        pipeline.train()


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30000)
    parser.add_argument('--c_sampled', type=int, default=20)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--shot_k', type=int, default=5)
    parser.add_argument('--query_k', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=None)
    opts = parser.parse_args()

    main(opts)
