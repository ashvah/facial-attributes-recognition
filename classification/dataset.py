#!/usr/bin/env python
# -*- coding:utf8 -*-

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import os


class Fig_data(Dataset):
    def __init__(self, label_file, feature_dir, attr_list=None, trans=None, train=True):
        self.label_file = label_file
        self.feature_dir = feature_dir
        self.train = train
        if self.label_file:
            with open(self.label_file, 'r') as f:
                self.len = int(f.readline())
                self.attribute_name = f.readline().strip().split()
                self.attr_num = len(attr_list) if attr_list else len(self.attribute_name)
                self.attr_list = list(map(lambda line: line.strip().split(), f))
                assert len(self.attr_list) == self.len
            if not attr_list:
                self.attr_need = [i for i in range(len(self.attribute_name))]
            else:
                self.attr_need = [self.attribute_name.index(a) for a in attr_list]
        else:
            self.fig_list = sorted(os.listdir(feature_dir))
            self.len = len(self.fig_list)

        if not trans:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=1, contrast=0.5, saturation=0.5),
                transforms.RandomAffine(degrees=0, scale=(.9, 1.1), shear=40, translate=(0, 0.1)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(p=0.1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            self.transform = trans

    def __getitem__(self, index):
        if self.train:
            img_name = self.attr_list[index][0]
            img_attr = np.array(list(map(lambda at: 1 if at == '1' else 0, self.attr_list[index][1:])))
            img_attr = img_attr[self.attr_need]
            assert len(img_attr) == self.attr_num
            img_data = self.transform(Image.open(self.feature_dir + '/' + img_name))
            return img_name, img_attr, img_data
        else:
            if not self.label_file:
                img_name = self.fig_list[index]
            else:
                img_name = self.attr_list[index][0]
            img_data = self.transform(Image.open(self.feature_dir + '/' + img_name))
            return img_name, img_data

    def __len__(self):
        return self.len


if __name__ == "__main__":
    dataset = Fig_data("../partition/train.txt", "../img_align_celeba", ['Eyeglasses', 'Male'])
    print(dataset.__len__())
    print(dataset.attribute_name)
    name, attr, img = dataset.__getitem__(0)
    print(name)
    print(attr)
    print(img)
    dataset = Fig_data("../partition/faces_valid.txt", "../faces-spring-2020", ['correct_label'])
    name, attr, img = dataset.__getitem__(1)
    print(name)
    print(attr)
    img.save("./1.jpg")
