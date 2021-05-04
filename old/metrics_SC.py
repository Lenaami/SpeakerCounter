import numpy as np
import argparse
import os
import sklearn
import time
import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import warnings
warnings.filterwarnings('ignore')

'''
args = {
    'v_vad': '2.0'
    'path_to_file_csv': 'outputs\\AMIcsv\\' # путь к файлам
    'path_to_file_rttm': 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_' + v_vad + '\\'
    'path_to': 'AMI_amp4mean_without0_old_rttm' + v_vad + '_array1-01' 
    'path_files': 'F:\\amicorpus_1'
    'SEC': 1
    'n_classes': 4 # или 5
}

'''

def get_labels_AMI(name, sec, useVAD = False):

    file_csv = 'outputs\\AMIcsv\\' + name + '.count.csv' # путь к файлам

    DATA = pd.read_csv(file_csv, delimiter=',') # колонки - start, duration, count или (start, end, duration, count)
    data = DATA.values
    data[:, 1] = data[:, 0] + data[:, 1] # переделываю в start, end, count

    count = []

    if useVAD == True:
        seconds = get_seconds_VAD(name, sec) # секунды, для которых нужно предсказать кол-во дикторов

        for i in range(len(seconds)):
            persons = [0]
            for st, ed, pers in data:
                if (seconds[i] + sec - st) > 0 and (ed - seconds[i]) > 0:
                    persons.append(pers)
            count.append(max(persons))
    else:
        # разметка по секундам целого файла
        for i in range(int(data[-1, 1] // sec + 1)):
            persons = [0]
            for st, ed, pers in data:
                if ((i+1)*sec - st) > 0 and (ed - i*sec) > 0:
                    persons.append(pers)
            count.append(max(persons))

    return count

def get_seconds_VAD(name, sec, useRTTM = True, inverseVAD = False): # inverse - взять те промежутки, где VAD НЕ отметил речь

    if useRTTM == True:
        # Использование VADа
        file_rttm = 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_' + v_vad + '\\' + name + '.segments.rttm'

        data = []
        with open(file_rttm, encoding='utf-8') as r_file:
            for line in r_file:
                ln = line.split(' ')
                data.append([float(ln[2]), float(ln[3])])b

    else:
        # Идеальная разметка
        file_csv = 'outputs\\AMIsilence\\' + name + '.no_silence.csv' # путь к файлам

        DATA = pd.read_csv(file_csv, delimiter=',') # колонки - start, duration
        data = DATA.values

   
    seconds = []

    # с начала (которое указано), по одной секунде, оставшийся отрезок меньше 0.1-0.2?, -> нужные секунды
    # пример 11.09, 4.4399999999999995 -> 11.09, 12.09, 13.09, 14.09, 15.09 --> далее перечисление продолжается со следующего промежутка

    '''
    for st, dur in data:
        times = float(dur) // sec
        if float(dur) % sec > float(sec) / 5:
            times += 1
        
        for i in range(int(times)):
            for_read.append(float(st) + i*sec)
    '''

    # Сравнение с общей разметкой по 1 секунде
    for st, dur in data:
        times = (dur + (st - int(st))) // sec + ((dur + (st - int(st))) % sec > 0) # вычисление кол-ва секунд, которые нужно обработать
        
        for i in range(times):
            seconds.append(int(st) + i*sec)


    seconds = list(set(seconds)) # убираем повторения (если есть)

    if inverseVAD == True:        
        seconds_all = list(np.arange(length[name])) # список секунд (нужно знать длину файлов)
        return [s for s in seconds_all if s not in seconds] # not in - участки без речи, in участки с речью

    return seconds  

def read_predict(name, )

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


    # CUSTOM CODE

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

    path_to = 'AMI_amp4mean_without0_old_rttm' + v_vad + '_array1-01'  # without0 - использование разметки csv с речевой активностью, old - секунды целочисленные, rttm0.0 - VAD Леши с порогом 0.0 (если нет - идеальная разметка), cl0-1 - классы
    #path_to = 'AMI_amp4mean_without0_old_array1-01'

    ###### !!!!!!!!!!!!! Унифицировать выход !!!!!!!!!!! - вроде сделано (почти - номер модели)

    if not os.path.exists('outputs\\' + args.model):
        os.makedirs('outputs\\' + args.model)
    
    if not os.path.exists('outputs\\' + args.model + '\\' + path_to):
        os.makedirs('outputs\\' + args.model + '\\' + path_to)

    #true_labels = get_labels_AMI('outputs\\AMIcsv\\' + 'EN2001a' + '.count.csv', SEC)
    #print(true_labels)

    t_stop = time.time()
    print("Prepare time: " + str(t_stop - t_start))
    t_start = time.time()
    
    names_audio = os.listdir('outputs\\AMIcsv') ########## ВЗЯТЬ ИМЕНА АУЙДИОФАЛОВ - сделано

    t_metrics = time.time()

    f1 = []

    true_l = []
    pred_l = []

    #print(length)
    #for name_au in names_audio:
    for name in name_f: # [:10] [70:80]

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI(name, SEC, useVAD = True)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)
        
        # ТОЛЬКО ДЛЯ МОДЕЛЕЙ 1-4
        
        pred_labels = []
        for i in range(len(pr_labels)):
            #pred_labels.append(pr_labels[i] + 1)
            pred_labels.append(pr_labels[i])
        #print(len(true_labels), len(pred_labels))
        

        true = [0]*max(len(pred_labels), len(true_labels))
        pred = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred[:len(pred_labels)] = pred_labels[:]

        ### Классификация для разного количества классов
        '''
        num_cl = 1 # до n+
    
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

    #indexs = [i for i in range(1, 11)]
    indexs = [i for i in range(0, 11)]

    cm = metrics.confusion_matrix(true, pred, normalize='pred') # 'true', 'pred', 'all'
    rg = cm.shape[0]

    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_precision.png')

    cm = metrics.confusion_matrix(true, pred, normalize='true') # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_recall.png')

    with open('outputs\\' + args.model + '\\' + path_to + '\\_report.txt', 'w') as f:
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