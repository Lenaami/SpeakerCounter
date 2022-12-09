# coding=utf-8

import os
import time
from tqdm import tqdm
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

from utils_.model import load_model
from utils_.data_loader import AmiDataset

def main(sec):

    # Список параметров
    with open('./yaml/train.yaml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args['sec'] = sec
    args['features'] = 'mfcc'
    args['model_arch'] = 'CNN'

    model, model_name = load_model(**args)

    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    lr = args['start_lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epoch = 1  # args['n_epochs']

    for epoch in range(n_epoch):
        model.train()
        train_loss = 0

        optimizer.zero_grad()
        X = np.random.sample((1, 1, int(100*args['sec']), 40)) # 201 , 128, 40
        test = torch.tensor(X).cuda().float()
        y = np.random.randint(0, 4, 1)
        labs = torch.tensor(y).cuda().long()
        outputs = model(test)
        #print(outputs.shape)
        #print(labs.shape)
        loss = criterion(outputs, labs)
        train_loss += loss.detach().cpu().numpy()
        print('Train_loss: ', train_loss)
        loss.backward()
        optimizer.step()



if __name__ == '__main__':

    sec = [0.5, 1, 1.5, 2, 3, 4, 5]
    for s in sec:
        main(s)