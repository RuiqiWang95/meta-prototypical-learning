# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  14:42

import os
import os.path as osp
import numpy as np
from PIL import Image
import csv
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


ROOT = 'dir/to/MiniImagenet'


class MiniImageNet(Dataset):
    """

    """
    def __init__(self, mode):
        """

        :param mode:
        """
        csv_path = osp.join(ROOT, mode+'.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data_file = []
        self.label = []
        lb = -1
        cls = []
        for l in lines:
            name, cid = l.split(',')
            path = osp.join(ROOT, 'images', name)
            if cid not in cls:
                cls.append(cid)
                lb += 1
            self.data_file.append(path)
            self.label.append(lb)

        self.transform = transforms.Compose([
            lambda x : Image.open(x).convert('RGB'),
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        file, label = self.data_file[index], self.label[index]
        data = self.transform(file)
        return data, label


class MiniImageNetSampler():
    """

    """
    def __init__(self, label, n_batch, n_way, k_sq):
        """

        :param label:
        :param n_batch:
        :param n_way:
        :param k_sq:
        """
        self.n_batch = n_batch
        self.n_way = n_way
        self.k_sq = k_sq
        self.cls_idx_pool = []

        label = np.array(label)

        for cls in range(max(label)+1):
            idxes = np.argwhere(label==cls).reshape(-1)
            idxes = torch.from_numpy(idxes)
            self.cls_idx_pool.append(idxes)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for idx_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.cls_idx_pool))[:self.n_way]
            for c in classes:
                cls_idxes = self.cls_idx_pool[c]
                pos = torch.randperm(len(cls_idxes))[:self.k_sq]
                batch.append(cls_idxes[pos])
            batch = torch.stack(batch).reshape(-1)
            yield batch


def get_dataloader(mode, task_num, n_way, shot_k, query_k, num_workers=8):
    dataset = MiniImageNet(mode)
    sampler = MiniImageNetSampler(dataset.label, task_num, n_way, shot_k+query_k)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == '__main__':

    trainset = MiniImageNet('train')
    train_sampler = MiniImageNetSampler(trainset.label, 100,
                                      20, 5+15)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    for i, batch in enumerate(train_loader):
        print(batch[0].size())
        print(batch[1])