# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  9:41

import torch
import torch.nn as nn
import torch.nn.functional as F

from convnet import Convnet

class protoPipe(nn.Module):
    """

    """
    def __init__(self, encoder, shot_k=5, query_k=15):
        """

        :param encoder:
        :param n_way:
        :param shot_k:
        :param query_k:
        """
        super().__init__()
        self.encoder = encoder
        self.s = shot_k
        self.q = query_k

    def forward(self, task, n_way):
        """

        :param task:
        :return:
        """
        feats = self.encoder(task).view(n_way, self.s+self.q,-1)
        shot_feats, query_feats = feats[:,:self.s,:].contiguous(), feats[:,self.s:,:].contiguous()
        cls_proto = shot_feats.mean(dim=1)
        query_feats = query_feats.view(n_way*self.q, -1)
        query_labels = torch.arange(n_way).view(-1,1).repeat(1,self.q).long().to(query_feats.device)

        cls_proto_ext = cls_proto.unsqueeze(0).repeat(n_way*self.q, 1, 1)
        query_feats_ext = query_feats.unsqueeze(1).repeat(1, n_way, 1)
        dis = torch.pow(cls_proto_ext-query_feats_ext, 2).sum(dim=2)
        relations = F.log_softmax(-dis, dim=1)
        # print(dis.mean())
        loss = -relations.gather(1, query_labels.view(-1,1)).mean()

        pred = torch.argmax(relations, dim=1)
        acc = (pred==query_labels.view(-1)).float().mean().item()

        # print((pred==query_labels.view(-1)).float().sum().item())

        return loss, acc


if '__main__' == __name__:
    from torch.utils.data import DataLoader
    from data import MiniImageNet, MiniImageNetSampler

    trainset = MiniImageNet('train')
    train_sampler = MiniImageNetSampler(trainset.label, 2,
                                        20, 5+15)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    convnet = Convnet().cuda()

    pipe = protoPipe(convnet)
    for epoch, batch in enumerate(train_loader, 0):
        print(epoch)
        batch = batch[0]
        loss, acc = pipe(batch.cuda(), 20)
        loss.backward()
        print(loss)