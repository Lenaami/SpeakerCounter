import numpy as np
import soundfile as sf
import argparse
import os
import keras
import sklearn
import librosa
from keras import backend as K
import time
import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

import warnings
warnings.filterwarnings('ignore')

path_audio = 'F:/amicorpus_1'
v_vad = '2.0'

def get_labels_AMI(name, sec, noSilence = False):

    #pd.read_csv(file, delimiter=',', names=['start', 'end', 'duration', 'count'])
    file_csv = 'outputs\\AMIcsv\\' + name + '.count.csv'

    data = []
    with open(file_csv, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter = ",")
        count = 0
        for row in file_reader:
            if count == 0:
                count += 1
            else:
                data.append([row[0], row[1], row[3]])
                count += 1
        
        # Файлик, где указаны границы активности / начало, длительность
        # отфильтровать (?) значения, которые предсказала сеть
        # нужно брать те секунды, котоорые попали в промежуток

    count = []

    if noSilence == True:
        seconds = get_VAD_output(name, sec)

        for i in range(len(seconds)):
            persons = [0]
            for st, ed, pers in data:
                if (seconds[i] + sec - float(st)) > 0 and (float(ed) - seconds[i]) > 0:
                    persons.append(int(pers))
            count.append(max(persons))
    else:
        for i in range(int(float(data[-1][1]) // sec + 1)):
            persons = [0]
            for st, ed, pers in data:
                if ((i+1)*sec - float(st)) > 0 and (float(ed) - i*sec) > 0:
                    persons.append(int(pers))
            count.append(max(persons))

    return count

def get_VAD_output(name, sec):

    # rttm    
    file_rttm = 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_' + v_vad + '\\' + name + '.segments.rttm'

    data = []
    with open(file_rttm, encoding='utf-8') as r_file:
        for line in r_file:
            ln = line.split(' ')
            data.append([float(ln[2]), float(ln[3])])
    
    for_read = []

    # Проверить на пересечения в секундах!!!!!!
    # Сравнение с изначальной разметкой (не по умному)
    for st, dur in data:
        times = (dur + (st - int(st))) // sec
        # times = (float(dur) + (float(st) - int(float(st)))) // sec
        if (dur + (st - int(st))) % sec > 0:
        # if (float(dur) + (float(st) - int(float(st)))) % sec > 0:
            times += 1
        
        for i in range(int(times)):
        #    for_read.append(int(float(st)) + i*sec)
            for_read.append(int(st) + i*sec)

    seconds_sp = list(set(for_read))

    seconds_all = []
    for i in range(0, seconds_sp[-1] + 1)
        if i in seconds_sp:
            seconds_all.append(1)
        else:
            seconds_all.append(0)

    #seconds = [sec for sec in seconds_all if sec not in seconds_sp] # not in - участки без речи, in участки с речью
    
    return seconds_all  


if __name__ == '__main__':

    t_start = time.time()

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


    path_f = []
    name_f = []
    for d, dirs, files in os.walk(args.audio): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            if f.split('.')[0] != 'ES2002d':
                path_f.append(path) # добавление адреса в список
                name_f.append(f.split('.')[0])

    #
    #predict = []
    #
    #speaker_count = []

    ############# ОТРЕЗОК СИГНАЛА
    SEC = 1
    #############

    # модель 0-4
    #path_to = 'AMI_amp4mean_without1_old_rttm' + v_vad + '_array1-01'  # withoun 0 - использование разметки csv с речевой активностью, old - секунды целочисленные, rttm0.0 - VAD Леши с порогом 0.0 (если нет - идеальная разметка)
    path_to = 'AMI_amp5_array1-01'

    save_to = 'outputs\\VAD+SC\\' + path_to + '_rttm' + v_vad

    ###### !!!!!!!!!!!!! Унифицировать выход !!!!!!!!!!! - вроде сделано (почти - номер модели)

    if not os.path.exists('outputs\\' + args.model):
        os.makedirs('outputs\\' + args.model)
    
    if not os.path.exists('outputs\\' + args.model + '\\' + path_to):
        os.makedirs('outputs\\' + args.model + '\\' + path_to)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    #true_labels = get_labels_AMI('outputs\\AMIcsv\\' + 'EN2001a' + '.count.csv', SEC)
    #print(true_labels)

    t_stop = time.time()
    print("Prepare time: " + str(t_stop - t_start))
    t_start = time.time()
    
    
    
    ################
    # Метрики
    ################
    
    names_audio = os.listdir('outputs\\AMIcsv') ########## ВЗЯТЬ ИМЕНА АУЙДИОФАЛОВ - сделано

    t_metrics = time.time()

    f1 = []

    true_l = []
    pred_l = []
    pred_v = []

    #for name_au in names_audio:
    for name in name_f: # [:10] [70:80]

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI(name, SEC, noSilence = False)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)
        
        pr_vad = get_VAD_output(name, SEC)

        # ТОЛЬКО ДЛЯ МОДЕЛЕЙ 1-4
        
        pred_labels = []
        for i in range(len(pr_labels)):
            #pred_labels.append(pr_labels[i] + 1)
            pred_labels.append(pr_labels[i])
        #print(len(true_labels), len(pred_labels))
        

        true = [0]*max(len(pred_labels), len(true_labels))
        pred_sc = [0]*max(len(pred_labels), len(true_labels))
        pred_vad = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred_sc[:len(pred_labels)] = pred_labels[:]
        pred_vad[:len(pr_vad)] = pr_vad[:]

        ### Классификация для разного количества классов
        
        num_cl = 1 # до n+
    
        for i in range(len(true)):
            if true[i] > num_cl:
                true[i] = num_cl
            if pred_sc[i] > num_cl:
                pred_sc[i] = num_cl
        
        ###

        true_l.extend(true)
        pred_l.extend(pred_sc)
        pred_v.extend(pred_vad)

        

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

    #indexs = [i for i in range(1, 11)]
    indexs = [i for i in range(0, 11)]

    cm = metrics.confusion_matrix(true, pred, normalize='pred') # 'true', 'pred', 'all'
    rg = cm.shape[0]

    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig(save_to + '\\_conf_matrix_precision.png')

    cm = metrics.confusion_matrix(true, pred, normalize='true') # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig(save_to + '\\_conf_matrix_recall.png')

    with open(save_to + '\\_report.txt', 'w') as f:
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

    t_stop = time.time()
    print("Time: " + str(t_stop - t_start))
    print("Metrics: " + str(t_stop - t_metrics))
    t_metrics = time.time()
#'''
    
'''
def eer_score(y_true, y_pred_probs): # y_pred как вероятности
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    fnr = 1 - tpr

    min_rate_idx = np.argmin(np.abs(fpr - fnr))
    eer_thr = thresholds[min_rate_idx]
    eer = np.mean([fpr[min_rate_idx], fnr[min_rate_idx]])
    return eer, eer_thr

EER 24

'''
