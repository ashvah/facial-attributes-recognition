# !/usr/bin/env python
# -*- coding:utf8 -*-

from .dataset import Fig_data
from .Xception import XceptionModel
from .CNN import CNNModel
from .utils import get_logger, get_transform
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import glob
import warnings
import argparse
import numpy as np

warnings.filterwarnings("ignore")

model_names = ['Xception', 'CNN']
models = [XceptionModel, CNNModel]
model_dict = dict([(k, v) for k, v in zip(model_names, models)])

parser = argparse.ArgumentParser()
parser.add_argument('-model', choices=model_names)
parser.add_argument('-optim', choices=['Adam', 'SGD'])
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-load', action='store_true')
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-parallel', action='store_true')
parser.add_argument('-l2', type=float, default=1e-8)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-num_of_workers', type=int, default=4)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-map', type=int, default=10)
args = parser.parse_args()

LR = args.lr
EPOCH = args.epoch
LOAD = args.load
MODEL_NAME = args.model
DATAPARA = args.parallel
MOMENTUM = args.momentum
NUM_OF_WORKERS = args.num_of_workers
OPTIM = args.optim
L2 = args.l2
BATCH_SIZE = args.batch_size
MAP = args.map

img_dir = '../img_align_celeba'
train_file = '../partition/hair_train.txt'
valid_file = '../partition/hair_valid.txt'
test_file = '../partition/hair_test.txt'
weight_dir = '../weights/' + MODEL_NAME
log_dir = '../log/' + MODEL_NAME
attr = ['gender', 'glass', 'hair']
num_of_classes = len(attr) if attr else 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    trans, _ = get_transform('celeb')
    train_data = Fig_data(train_file, img_dir, attr)
    valid_data = Fig_data(valid_file, img_dir, attr, trans=trans)
    VALID_SIZE = len(valid_data)

    logger = get_logger(log_dir + '/train_log', name='train')

    if LOAD:
        wfn = sorted(glob.glob(weight_dir + '/weight*'))[-1]
        logger.info('loaded weights file: %s' % wfn)
        model = model_dict.get(MODEL_NAME)(num_of_classes)
        model.load_state_dict(torch.load(wfn))
    else:
        model = model_dict.get(MODEL_NAME)(num_of_classes)

    if DATAPARA:
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)
    model.train()

    loss_f = torch.nn.BCELoss()
    loss_f = loss_f.to(DEVICE)

    if OPTIM == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=L2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    logger.info("Optimizer: " + str(type(optimizer)))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS)
    data_amount = len(train_data)

    logger.info('Start training!')
    logger.info('Using ' + DEVICE.type)

    for e in range(EPOCH):
        aver_loss = 0
        for i, train_data in enumerate(train_loader):
            _, label_t, feature_t = train_data
            feature_t, label_t = feature_t.to(DEVICE), label_t.to(DEVICE)
            feature_t, label_t = feature_t.float(), label_t.float()

            try:
                y_pred = model.forward(feature_t)
                loss_t = loss_f(y_pred, label_t)
                aver_loss += loss_t.item() * feature_t.shape[0]
                logger.info(
                    "Epoch {}: [{}/{}], training set loss: {:.4}".format(e, i * BATCH_SIZE, data_amount, loss_t.item()))
                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logger.error("WARNING: out of memory")
                    del feature_t, label_t
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise exception

            del feature_t, label_t

        aver_loss = aver_loss / data_amount
        logger.info("Epoch {}: aver loss: {:.4}".format(e, aver_loss))

        aver_loss_v = 0.0
        aver_map_at_n = 0.0
        model.eval()
        with torch.no_grad():
            for valid_data in valid_loader:
                _, label_v, feature_v = valid_data
                label_v, feature_v = label_v.to(DEVICE).float(), feature_v.to(DEVICE).float()
                y_v = model.forward(feature_v)
                loss_v = loss_f(y_v, label_v)
                aver_loss_v += loss_v.item() * feature_v.shape[0]
                # g = MAP_at_n(y_v.cpu().numpy(), label_v.cpu().numpy(), MAP)
                g = average_precision_score(label_v.cpu().numpy().reshape(-1), y_v.cpu().numpy().reshape(-1))
                if np.isnan(g):
                    g = 0.0
                logger.info("Epoch {}: MAP@{}: {:.4}".format(e, MAP, g))
                aver_map_at_n += g * feature_v.shape[0]
        aver_loss_v = aver_loss_v / VALID_SIZE
        aver_map_at_n = aver_map_at_n / VALID_SIZE
        logger.info(
            "Epoch {}: validation set loss: {:.4} MAP@{} {:.4}".format(e, aver_loss_v, MAP, aver_map_at_n))
        torch.save(model.state_dict(), weight_dir + '/weight%.4f_%d_%dmodel.pkl' % (aver_map_at_n, e, i * BATCH_SIZE))
        # torch.save(model, weight_dir + '/weight%.4f_%d_%dmodel.pkl' % (aver_map_at_n, e, i * BATCH_SIZE))
        model.train()

    logger.info('Finish training!')
    logger.info('Start Testing!')
    test_data = Fig_data(test_file, img_dir, attr, trans=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS)
    TEST_SIZE = len(test_data)

    logger.info("Test set loss: ")
    model.eval()
    aver_loss_t = 0.0
    aver_map_at_n = 0.0
    with torch.no_grad():
        for test_data in test_loader:
            _, label_t, feature_t = test_data
            label_t, feature_t = label_t.to(DEVICE).float(), feature_t.to(DEVICE).float()
            y_t = model.forward(feature_t)
            loss_t = loss_f(y_t, label_t)
            aver_loss_t += loss_t.item() * feature_t.shape[0]
            g = average_precision_score(label_t.cpu().numpy(), y_t.cpu().numpy())
            logger.info("MAP@{}: {:.4}".format(MAP, g))
            aver_map_at_n += g * feature_t.shape[0]

    aver_loss_t = aver_loss_t / TEST_SIZE
    aver_map_at_n = aver_map_at_n / TEST_SIZE
    logger.info("Average test set loss: {:.4} MAP@{}: {:.4}".format(aver_loss_t, MAP, aver_map_at_n))
    logger.info("End testing")


if __name__ == "__main__":
    train()
