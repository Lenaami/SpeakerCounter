#############
# Команда для запуска
# python predict_metrics.py F:/1amicorpus/loud --model 5
# python predict_metrics.py f:/amicorpus_1 --model 8
#############

import numpy as np
import soundfile as sf
import argparse
import os
#import keras
import sklearn
import librosa
#from keras import backend as K

import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

eps = np.finfo(np.float).eps


def get_labels_AMI(file, sec):

    #pd.read_csv(file, delimiter=',', names=['start', 'end', 'duration', 'count'])

    data = []
    with open(file, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter = ",")
        count = 0
        for row in file_reader:
            if count == 0:
                count += 1
            else:
                data.append([row[0], row[1], row[3]])
                #print(f'    {row[0]} - {row[1]} и он родился в {row[2]} году.')
                count += 1
        #print(f'Всего в файле {count} строк.')
    
    count = []

    for i in range(int(float(data[-1][1]) // sec + 1)):
        persons = [0]
        for st, ed, pers in data:
            if ((i+1)*sec - float(st)) > 0 and (float(ed) - i*sec) > 0:
                persons.append(int(pers))
        count.append(max(persons))

    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )

    parser.add_argument(
        'audio',
        help='audio file (samplerate 16 kHz) of 5 seconds duration'
    )

    parser.add_argument(
        '--model', default='CRNN',
        help='model name'
    )

    args = parser.parse_args()


    # CUSTOM CODE

    path_f = []
    name_f = []
    for d, dirs, files in os.walk(args.audio): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('.')[0])


    ############# ОТРЕЗОК СИГНАЛА
    SEC = 1
    #############

    path_to = 'AMI_amp5_array1-01'

    add_name = '_without0' # _klasses0-2 _without0

    ################
    # Метрики
    ################
    
    f1 = []

    true_l = []
    pred_l = []

    for name in name_f:

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI('outputs\\AMIcsv\\' + name + '.count.csv', SEC)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pred_labels = pickle.load(f)
        
        ### Исключение нулевого класса (вместо выравнивания по длине)
        
        true = []
        pred = []
        #print(len(true_labels), len(pred_labels))
        for i in range(min(len(true_labels), len(pred_labels))):
            if true_labels[i] != 0:
                true.append(true_labels[i])
                pred.append(pred_labels[i])
        

        ### Выравнивание по длине (заполняем нулями пустоту)
        '''
        true = [0]*max(len(pred_labels), len(true_labels))
        pred = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred[:len(pred_labels)] = pred_labels[:]
        '''
        
        ### Классификация для разного количества классов
        '''
        num_cl = 2 # до n+
    
        for i in range(len(true)):
            if true[i] > num_cl:
                true[i] = num_cl
            if pred[i] > num_cl:
                pred[i] = num_cl
        '''
        ###

        true_l.extend(true)
        pred_l.extend(pred)

        f1.append(f1_score(true, pred, average='weighted'))

    print('Среднее по датасету')
    print(np.mean(f1))

    print('Объединенные результаты по датасету')
    print(f1_score(true_l, pred_l, average='weighted'))

    print(metrics.classification_report(true_l, pred_l))

    print('Confusion matrix (Precision)')
    print(metrics.confusion_matrix(true_l, pred_l, normalize='pred'))  # 'true' - recall, 'pred' - precisoin, 'all'
    print('Confusion matrix (Recall)')
    print(metrics.confusion_matrix(true_l, pred_l, normalize='true'))

    indexs = [i for i in range(11)]

    cm = metrics.confusion_matrix(true, pred, normalize='pred') # 'true', 'pred', 'all'
    rg = cm.shape[0]

    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_precision' + add_name + '.png')

    cm = metrics.confusion_matrix(true, pred, normalize='true') # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_recall' + add_name + '.png')

    with open('outputs\\' + args.model + '\\' + path_to + '\\_report' + add_name + '.txt', 'w') as f:
        f.write('Файлы: ' + args.audio + '\n')
        f.write('Модель: ' + args.model + '\n')
        f.write('\n')
        f.write('F1-score (weighted)\n')
        f.write('Среднее по датасету: ' + str(np.mean(f1)) + '\n')
        f.write('Объединенные результаты по датасету: ' + str(f1_score(true_l, pred_l, average='weighted')) + '\n')
        f.write('\n')
        f.write(str(metrics.classification_report(true_l, pred_l)) + '\n')
        f.write('\n')
        f.write('Confusion matrix (Precision)\n')
        f.write(str(metrics.confusion_matrix(true_l, pred_l, normalize='pred')) + '\n')
        f.write('Confusion matrix (Recall)\n')
        f.write(str(metrics.confusion_matrix(true_l, pred_l, normalize='true'))+ '\n')

#'''






