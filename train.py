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
import torch.nn.functional as F

from utils_.model import load_model
from utils_.data_loader import AmiDataset
from utils_.metrics import get_labels_AMI

def eval_model(model, eval_dataset):
    model.eval()
    forecast, true_labs = [], []
    with torch.no_grad():
        for wavs, labs in tqdm(eval_dataset):
            wavs, labs = wavs.cuda().float(), labs.detach().numpy()
            true_labs.append(labs)
            outputs = model(wavs) #model(wavs) #model.inference(wavs)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.detach().cpu().numpy().argmax(axis=1)
            forecast.append(outputs)
    forecast = [x for sublist in forecast for x in sublist]
    true_labs = [x for sublist in true_labs for x in sublist]
    return f1_score(forecast, true_labs, average='macro')


def train(sec, features):

    # Список параметров
    with open('./yaml/train.yaml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Изменяемые параметры
    bs = int(args['batch_size'] // sec)
    args['batch_size'] = bs
    args['sec'] = sec
    args['features'] = features

    # Сеты AMI
    with open(args['set_path']) as f:
        ami_sets = yaml.load(f, Loader=yaml.FullLoader)

    #class_weights = [0.137, 0.705, 0.135, 0.02, 0.002]
    #dataset = AmiDataset(ami_sets['train'], **args)
    #weights = np.array([class_weights[int(label)] for label in dataset.labels])
    #weights = torch.from_numpy(weights).double()
    #weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset.labels))

    # Загружаем train датасет
    if args['dataset'] == 'AMIaug':
        args['augmented']['sec'] = sec
        args['augmented']['features'] = features

        dataset = AmiDataset(ami_sets['train'], **args['augmented']) # Если нужна аугментация **args['augmented'], если нет **args

    else:
        dataset = AmiDataset(ami_sets['train'], **args)

    train_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'], shuffle=False, num_workers=8#, sampler=weighted_sampler
    )

    # Загружаем тестовый датасет
    test_loader = DataLoader(
        AmiDataset(ami_sets['test'], **args),
        batch_size=args['batch_size'], shuffle=False, num_workers=8
    )

    # Загрузка/создание модели
    model, model_name = load_model(**args)
    
    s_model_save_dir = args['checkpoints_path'] + model_name

    s_log_file = s_model_save_dir + "/model_log.csv" # xlsx

    with open(s_log_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_f1 (macro)'])
        #writer.writerow(['epoch', 'test_f1'])

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time()

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.L1Loss()
    #model = nn.DataParallel(model)
    model = model.cuda()
    lr = args['start_lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epoch = args['n_epochs']
    best_f1 = 0
    for epoch in range(n_epoch):
        model.train()
        #train_loss = 0
        train_loss = []
        for wavs, labs in tqdm(train_loader):
            optimizer.zero_grad()
            wavs, labs = wavs.cuda().float(), labs.cuda().long()
            outputs = model(wavs)
            labs = torch.squeeze(labs, 1)
            #print(outputs.shape)
            #print(labs.shape)
            loss = criterion(outputs, labs)
            #train_loss += loss.detach().cpu().numpy()
            train_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
        f1 = eval_model(model, test_loader)
        #f1_train = eval_model(model, train_loader)
        #print(f'epoch: {epoch}, f1_train: {f1_train}, f1_test: {f1}')
        #print(f'epoch: {epoch}, loss: {train_loss}, f1_test: {f1}')
        print(f'epoch: {epoch}, loss: {np.mean(np.asarray(train_loss))}, f1_test: {f1}')
        with open(s_log_file, 'a') as f:
            writer = csv.writer(f)
            #writer.writerow([epoch, f1])
            writer.writerow([epoch, np.mean(np.asarray(train_loss)), f1])
        if epoch < 10:
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), s_model_save_dir + '/model_' + str(epoch) + '_best.pt')
        if epoch == 10: best_f1 = 0
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), s_model_save_dir + '/model.pt')
        torch.save(model.state_dict(), s_model_save_dir + '/model_' + str(epoch) + '.pt')

        lr = lr * 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    t_stop = time.time()
    print("Training time : " + str(t_stop - t_start))


def main():

    #sec = [5, 4, 3, 2, 1.5, 1, 0.5]
    #sec = [5, 4, 3, 2]
    #sec = [5, 3, 4]
    #sec = [1, 2]
    #sec = [1.5]
    #features = ['stft', 'melsp', 'mfcc']
    #features = ['stft', 'mfcc']
    '''
    for s in sec:
        for feat in features:
            #print('cool')
            train(s, feat)
    '''
    train(0.5, 'stft')
    #train(1, 'mfcc')

if __name__ == '__main__':
    main()
