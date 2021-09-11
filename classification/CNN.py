#!/usr/bin/env python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torchvision


class SeparableConv(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, use_bias):
        super(SeparableConv, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module('conv1',
                              nn.Conv2d(inChannels, inChannels, kernel_size=kernel_size, groups=inChannels, padding=1,
                                        bias=use_bias))
        self.block.add_module('conv2', nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=use_bias))

    def forward(self, x):
        return self.block(x)


class xception_block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(xception_block, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module('separable_conv1', SeparableConv(inChannels, outChannels, 3, False))
        self.block.add_module('bn1', nn.BatchNorm2d(outChannels))
        self.block.add_module('relu1', nn.ReLU())
        self.block.add_module('separable_conv2', SeparableConv(outChannels, outChannels, 3, False))
        self.block.add_module('bn2', nn.BatchNorm2d(outChannels))
        self.block.add_module('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.res = nn.Sequential()
        self.res.add_module('conv1', nn.Conv2d(inChannels, outChannels, 1, stride=2))
        self.res.add_module('bn1', nn.BatchNorm2d(outChannels))

    def forward(self, x):
        out = self.block(x)
        res = self.res(x)
        return res + out


class Bottleneck(nn.Module):
    def __init__(self, nChannels, interChannels):
        super(Bottleneck, self).__init__()
        self.bottleneck = torch.nn.Sequential()
        self.bottleneck.add_module('bn1', torch.nn.BatchNorm2d(nChannels))
        self.bottleneck.add_module('relu1', torch.nn.ReLU())
        self.bottleneck.add_module('conv1', torch.nn.Conv2d(nChannels, interChannels, 1, bias=False))
        self.bottleneck.add_module('bn2', torch.nn.BatchNorm2d(interChannels))
        self.bottleneck.add_module('relu2', torch.nn.ReLU())
        self.bottleneck.add_module('conv2',
                                   torch.nn.Conv2d(interChannels, interChannels, kernel_size=3, padding=1, bias=False))
        self.bottleneck.add_module('bn3', torch.nn.BatchNorm2d(interChannels))
        self.bottleneck.add_module('relu3', torch.nn.ReLU())
        self.bottleneck.add_module('conv3', torch.nn.Conv2d(interChannels, nChannels, 1, bias=False))

    def forward(self, x):
        out = self.bottleneck(x)
        return x + out


class CNNModel(nn.Module):
    def __init__(self, num_of_classes):
        super(CNNModel, self).__init__()
        self.init = torch.nn.Sequential()
        self.init.add_module("conv1", torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False, padding=1))
        self.init.add_module("bn1", torch.nn.BatchNorm2d(32))
        self.init.add_module("relu1", torch.nn.ReLU())
        self.init.add_module("conv2", torch.nn.Conv2d(32, 64, 3, bias=False))
        self.init.add_module("bn2", torch.nn.BatchNorm2d(64))
        self.init.add_module("relu2", torch.nn.ReLU())

        self.intermediate = torch.nn.Sequential()

        last_channels = 64
        block_data = [128, 256, 512, 1024]
        for i in range(len(block_data)):
            self.intermediate.add_module("intermedate" + str(i), xception_block(last_channels, block_data[i]))
            last_channels = block_data[i]

        # 1024*7*7
        self.output = torch.nn.Sequential()
        self.output.add_module("linear", nn.Linear(1024 * 6 * 7, num_of_classes))
        self.output.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        init = self.init(x)
        mid = self.intermediate(init)
        mid = mid.reshape(-1, 1024 * 6 * 7)
        output = self.output(mid)
        return output


if __name__ == "__main__":
    a = torch.randn((2, 3, 218, 178))
    with torch.no_grad():
        result = CNNModel(2).forward(a).numpy()
    classifier = torchvision.models.resnet18(pretrained=True)
    with torch.no_grad():
        result = classifier.forward(a).numpy()
    print(result.shape)
