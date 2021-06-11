# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  10:19

import os.path as osp
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='metanet')
parser.add_argument("--name",type=str,default='config_tmp')
opts = parser.parse_args()

metanet_5 = [
        ('conv2d', [64, 3, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [5, 64 * 5 * 5]),
    ]

metanet_5_ori = [
        ('conv2d', [64, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [5, 64 * 5 * 5]),
    ]

metanet_10 = [
        ('conv2d', [64, 3, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [10, 64 * 5 * 5]),
    ]

metanet_10_ori = [
        ('conv2d', [64, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [10, 64 * 5 * 5]),
    ]

metaprotonet = [
        ('conv2d', [64, 3, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
]

config = locals()[opts.config]
print(config)
torch.save(config, osp.join('Config', opts.name))
