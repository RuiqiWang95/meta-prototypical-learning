# -*-coding : utf-8 -*-
# @Author   : Ruiqi Wang
# @time     :  20:08

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Convnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = conv_block(3,64)
        self.block2 = conv_block(64,64)
        self.block3 = conv_block(64,64)
        self.block4 = conv_block(64,64)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x.view(x.size(0), -1)


if '__main__' == __name__:
    m = Convnet().to('cuda')
    x = torch.randn(4,3,84,84).to('cuda')
    y = m(x)
    print(y.size())





