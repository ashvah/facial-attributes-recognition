#!/usr/bin/env python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch
import torchvision


class DenseModel(nn.Module):
    def __init__(self, num_of_classes):
        super(DenseModel, self).__init__()
        self.num_of_classes = num_of_classes
        self.model = torchvision.models.densenet121(pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        fc_inputs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_of_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.model(x)
        return res


if __name__ == "__main__":
    a = torch.randn((2, 3, 218, 178))
    with torch.no_grad():
        result = DenseModel(2).forward(a).numpy()
    print(result.shape)
    print(torchvision.models.densenet161(pretrained=True))
