#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
from multiprocessing import Pool
import logging
from .resnet import ResModel
from .resnet3 import Res3Model
from .densenet import DenseModel
from torchvision.transforms import transforms
import torch
import glob

model_names = ['Res', 'Res3']
models = [ResModel, Res3Model]
model_dict = dict([(k, v) for k, v in zip(model_names, models)])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, feature, mean=False, device=DEVICE):
    model.eval()
    with torch.no_grad():
        feature = feature.to(device).float()
        y = model.forward(feature).cpu().numpy()
    y = np.where(y == np.nan, np.zeros_like(y), y)
    if mean:
        y = np.mean(y, axis=0)
    # y = np.where(y >= 0.5, np.ones_like(y), np.zeros_like(y))
    return y


def load_model(path, device=DEVICE, name='CNN', num_of_classes=2, logger=None, datapara=False):
    wfn = sorted(glob.glob(path + '/weight*'))[-1]
    print(wfn)
    if logger:
        logger.info('loaded weights file: %s' % wfn)
    model = model_dict.get(name)(num_of_classes)
    model.load_state_dict(torch.load(wfn, map_location=DEVICE))
    if datapara:
        model = torch.nn.DataParallel(model)
    print(device)
    model = model.to(device)
    return model


def get_transform(dataset):
    if dataset == 'lfw_5590':
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomVerticalFlip(0.2),
            # transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        img_dir = "../MTFL/lfw_5590"
    else:
        trans = transforms.Compose([
            transforms.Resize((218, 178)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        img_dir = '../img_align_celeba'
    return trans, img_dir


def get_logger(filename, verbosity=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def ap_at_n(data, n=10):
    predictions, actuals = data
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)
    ap = 0.0
    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:  # in label prediction, the num should be 1 or 0
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def MAP_at_n(pred, actual, n=10):
    lst = zip(list(pred), list(actual))

    with Pool() as pool:
        all_ = pool.map(ap_at_n, lst, n)

    return np.mean(all_)
