#!/usr/bin/env python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch
import torchvision

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OutputBlock(nn.Module):
    def __init__(self):
        super(OutputBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class Res1Model(nn.Module):
    def __init__(self, num_of_classes):
        super(Res1Model, self).__init__()
        self.num_of_classes = num_of_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.glass = OutputBlock()
        for param in self.glass.parameters():
            param.requires_grad = False
        self.hair = OutputBlock()

    def forward(self, x):
        res = self.model(x)
        out_list = [self.glass(res), self.hair(res)]
        out = torch.cat(out_list, dim=1)
        return out


if __name__ == "__main__":
    a = torch.randn((2, 3, 218, 178))
    net = Res1Model(2)
    with torch.no_grad():
        result = net.forward(a)
    print(result[:, 0].reshape(-1, 1).shape)
