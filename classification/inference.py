# !/usr/bin/env python
# -*- coding:utf8 -*-

from .dataset import Fig_data
from .Xception import XceptionModel
from .CNN import CNNModel
from .utils import get_logger, predict, load_model
import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import argparse

warnings.filterwarnings("ignore")

model_names = ['Xception', 'CNN', 'CNN_norm', 'CNN_nohair', 'CNN_hairglass']
models = [XceptionModel, CNNModel]
model_dict = dict([(k, v) for k, v in zip(model_names, models)])

parser = argparse.ArgumentParser()
parser.add_argument('-model', choices=model_names)
parser.add_argument('-parallel', action='store_true')
parser.add_argument('-num_of_workers', type=int, default=4)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-map', type=int, default=10)
parser.add_argument('-dataset', choices=['lfw_5590', 'celeb'])
args = parser.parse_args()

MODEL_NAME = args.model
DATAPARA = args.parallel
NUM_OF_WORKERS = args.num_of_workers
BATCH_SIZE = args.batch_size
DATASET = args.dataset
MAP = args.map

train_file = '../partition/train.txt'
valid_file = '../partition/valid.txt'
test_file = '../partition/test.txt' if DATASET == 'celeb' else None
weight_dir = '../weights/' + MODEL_NAME
log_dir = '../result'
attr = ['gender', 'glass', 'hair']

num_of_classes = len(attr) if attr else 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def predict_all():
    logger = get_logger(log_dir + '/inference_log', name='inference', mode='w')

    model = load_model(weight_dir, DEVICE, logger, DATAPARA)

    df = pd.DataFrame.from_dict({'Name': [], 'Eyeglasses': [], 'Male': []})
    filename = log_dir + '/result.txt'
    df.to_csv(filename, header=True, index=False, columns=['Name', 'Eyeglasses', 'Male'], mode='w')

    logger.info('Start Testing!')
    trans, img_dir = get_transform(DATASET)
    test_data = Fig_data(test_file, img_dir, trans=trans, train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS)

    model.eval()
    for test_data in test_loader:
        name_t, feature_t = test_data
        y_t = predict(model, feature_t)
        df = pd.DataFrame.from_dict({'Name': list(name_t), 'Eyeglasses': y_t[:, 0], 'Male': y_t[:, 1]})
        df.to_csv(filename, header=False, index=False, columns=['Name', 'Eyeglasses', 'Male'], mode='a')

    logger.info("End testing")


if __name__ == "__main__":
    predict_all()
