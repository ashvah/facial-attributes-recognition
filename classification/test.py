# !/usr/bin/env python
# -*- coding:utf8 -*-

from .dataset import Fig_data
from .Xception import XceptionModel
from .CNN import CNNModel
from .utils import get_logger, predict, load_model
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import warnings
import argparse

warnings.filterwarnings("ignore")

model_names = ['Xception', 'CNN', 'CNN_nohair', 'CNN_norm', 'CNN_hairglass']
models = [XceptionModel, CNNModel]
model_dict = dict([(k, v) for k, v in zip(model_names, models)])

parser = argparse.ArgumentParser()
parser.add_argument('-model', choices=model_names)
parser.add_argument('-parallel', action='store_true')
parser.add_argument('-num_of_workers', type=int, default=4)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-map', type=int, default=10)
args = parser.parse_args()

MODEL_NAME = args.model
DATAPARA = args.parallel
NUM_OF_WORKERS = args.num_of_workers
BATCH_SIZE = args.batch_size
MAP = args.map

img_dir = '../img_align_celeba'
train_file = '../partition/train.txt'
valid_file = '../partition/valid.txt'
test_file = '../partition/test.txt'
weight_dir = '../weights/' + MODEL_NAME
log_dir = '../result'
attr = ['gender', 'glass', 'hair']
num_of_classes = len(attr) if attr else 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test():
    logger = get_logger(log_dir + '/test_log', name='test', mode='w')

    model = load_model(weight_dir, DEVICE, logger)

    logger.info('Start Testing!')
    test_data = Fig_data(test_file, img_dir, attr)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS)
    TEST_SIZE = len(test_loader)

    model.eval()
    aver_map_at_n = 0
    for test_data in test_loader:
        _, label_t, feature_t = test_data
        y_t = predict(model, feature_t)
        g = average_precision_score(label_t.float().numpy(), y_t)
        logger.info("MAP@{}: {:.4}".format(MAP, g))
        aver_map_at_n += g * feature_t.shape[0]

    aver_map_at_n = aver_map_at_n / TEST_SIZE
    logger.info("Average test set loss: MAP@{}: {:.4}".format(MAP, aver_map_at_n))
    logger.info("End testing")


if __name__ == "__main__":
    test()
