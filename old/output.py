#############
# Команда для запуска
# python output.py f:/amicorpus_1 --model 10
#############

import numpy as np
import soundfile as sf
import argparse
import os
import sklearn
import librosa

import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

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
        seconds = get_silence(name, sec)

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

def get_silence(name, sec):
    '''
    file_csv = 'outputs\\AMIsilence\\' + name + '.no_silence.csv'

    data = []
    with open(file_csv, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter = ",")
        count = 0
        for row in file_reader:
            if count == 0:
                count += 1
            else:
                data.append([float(row[0]), float(row[1])])
                count += 1
    '''
    # rttm

    file_rttm = 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_' + v_vad + '\\' + name + '.segments.rttm'

    data = []
    with open(file_rttm, encoding='utf-8') as r_file:
        for line in r_file:
            ln = line.split(' ')
            data.append([float(ln[2]), float(ln[3])])

    for_read = []

    # Сравнение с изначальной разметкой с целыми секундами (не по умному)
    for st, dur in data:
        times = (dur + (st - int(st))) // sec
        if (dur + (st - int(st))) % sec > 0:
            times += 1
        
        for i in range(int(times)):
            for_read.append(int(st) + i*sec)

    return list(set(for_read))    


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

    path_f = []
    name_f = []
    for d, dirs, files in os.walk(args.audio): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            if f.split('.')[0] != 'ES2002d':
                path_f.append(path) # добавление адреса в список
                name_f.append(f.split('.')[0])

    ############# ОТРЕЗОК СИГНАЛА
    SEC = 1
    #############

    #path_to = 'AMI_amp4mean_array1-01'
    path_to = 'AMI_amp4mean_without0_old_rttm' + v_vad + '_array1-01'

    ###### !!!!!!!!!!!!! Унифицировать выход !!!!!!!!!!! - вроде сделано (почти - номер модели)

    if not os.path.exists('outputs\\SCoutput\\' + args.model + '\\' + path_to):
        os.makedirs('outputs\\SCoutput\\' + args.model + '\\' + path_to)
    
    #if not os.path.exists('outputs\\' + args.model + '\\' + path_to):
    #    os.makedirs('outputs\\' + args.model + '\\' + path_to)



    #for name_au in names_audio:
    for name in name_f:

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)
        

        # Для тех моделей, где используется VAD
        
        seconds = get_silence(name, SEC)
        pred_labels = []
        for i in range(seconds[-1] + 1):
            if i in seconds:
                pred_labels.append(pr_labels[seconds.index(i)] + 1) # для модели 1-4 диктора
            else:
                pred_labels.append(0)
        

        # Для всех данных (учитывать модельку 0-4, 1-4)
        '''
        pred_labels = []
        for i in range(len(pr_labels)):
            pred_labels.append(pr_labels[i] + 1)  
            #pred_labels.append(pr_labels[i])
        '''   


        output = []
        for i in range(len(pred_labels)):
            output.append([i, SEC, pred_labels[i]])


        OUT = pd.DataFrame(output, columns=['start', 'duration', 'count'])
        #OUT = pd.DataFrame(output, columns=['start', 'end', 'speakers'])
        OUT.to_csv('outputs\\SCoutput\\' + args.model + '\\' + path_to + '\\' + name + '.SCout.csv', index=False)








