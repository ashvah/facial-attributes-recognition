#!/usr/bin/env python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch
import torchvision


class OutputBlock(nn.Module):
    def __init__(self):
        super(OutputBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class ResModel(nn.Module):
    def __init__(self, num_of_classes):
        super(ResModel, self).__init__()
        self.num_of_classes = num_of_classes
        self.model = torchvision.models.resnet34(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.output = OutputBlock()

    def forward(self, x):
        res = self.model(x)
        out = self.output(res)
        return out


if __name__ == "__main__":
    a = torch.randn((2, 3, 224, 224))
    with torch.no_grad():
        result = ResModel(2).forward(a).numpy()
    print(result.shape)
