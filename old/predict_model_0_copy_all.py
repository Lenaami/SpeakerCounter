#############
# Команда для запуска
# python predict_model.py F:/1amicorpus/loud --model 5
# python predict_model_0_copy_all.py f:/amicorpus_1 --model 11
#############

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

eps = np.finfo(np.float).eps

path_audio = 'F:/amicorpus_1'
v_vad = '4.0'

#file_csv = 'outputs\\AMIcsv\\'
#file_rttm = 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_1.0\\'
#SEC = 1
#path_to = 'AMI_amp4mean_without0_old_rttm10_array1-01' 
#model (номер)
#classes (4/5)

##### Чек-лист #####
# ПУТЬ
# МОДЕЛЬ 0-4 или 1-4
# rttm или идеальная разметка
# как получаются секунды
# сколько файлов обрабатывается (звук и разметка)
# исключили лишнюю запись? для rttm

def AMP_audio(audio):

    seconds = len(audio)//rate 

    AMP = 0
    X_max = []
    for i in range(seconds):
        #estimate.append(count(audio[i*rate*SEC:(i+1)*rate*SEC], model, scaler, SEC))
        X = np.abs(librosa.stft(audio[i*rate:(i+1)*rate], n_fft=400, hop_length=160)).T
        X_max.append(np.max(X))
        #if np.max(X) < 5 and AMP < np.max(X):
        #    AMP = np.max(X)

    if np.mean(X_max) < 4:
        AMP = 4/(np.mean(X_max) + 0.001)
    else:
        AMP = 1

    return AMP

def count(audio, model, scaler, sec, AMP):

    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    #print(audio)
    #print(audio.shape)
    X_0 = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    ####### ТУПОЕ УСИЛЕНИЕ АМПЛИТУДЫ
    '''
    if np.max(X_0) > 5:
        AMP = 1    
    else:
        AMP = AMP
        #AMP = 5/(np.max(X_0) + 0.001) # Поиграть с этой величиной / попробовать умножать на статичный коэф. (4?)
    '''
    X_0 *= AMP
    
    #######

    # apply global (featurewise) standardization to mean1, var0
    X_0 = scaler.transform(X_0)

    # cut to input shape length (500 frames x 201 STFT bins)
    X_0 = X_0[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!
    X = np.zeros((100, 201))

    X[:X_0.shape[0], :] = X_0 

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    #print(X.shape)

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]

def count_part(audio, rate, seconds, model, scaler, sec, AMP):

    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    #print(audio)
    #print(audio.shape)
    X_part = []
    #print(seconds)

    for i in range(len(seconds)):
        sec_begin = int(seconds[i]*rate)
        audio_part = audio[sec_begin:sec_begin+int(sec*rate)]
        #print(sec_begin, sec_begin+int(sec*rate))
        #print(audio_part.shape)
        
        X_0 = np.abs(librosa.stft(audio_part, n_fft=400, hop_length=160)).T

        X_0 *= AMP
    
    #######

    # apply global (featurewise) standardization to mean1, var0
        X_0 = scaler.transform(X_0)

    # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = X_0[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!
        X = np.zeros((int(100*sec), 201))

        X[:X_0.shape[0], :] = X_0 

    # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        X_part.append(X)

    # add sample dimension
    #X = X[np.newaxis, ...]
    
    X_part = np.asarray(X_part)

    if len(model.input_shape) == 4:
        X_part = X_part[:, np.newaxis, ...]

    #print(X_part.shape)

    ys = model.predict(X_part, verbose=0)
    #print(np.argmax(ys, axis=1))
    return np.argmax(ys, axis=1)

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

    return list(set(for_read))    
    

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

    ######## !!!!!!!! Переделать названия моделей  !!!!!!!!
    # load model
    model = keras.models.load_model(
        os.path.join('checkpoints\\' + args.model, 'the_network.h5'),
        #custom_objects={
        #    'class_mae': class_mae,
        #    'exp': K.exp
        #}
    )

    # print model configuration
    model.summary()
    # save as svg file
    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    # CUSTOM CODE

    path_f = []
    name_f = []
    for d, dirs, files in os.walk(args.audio): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            #if f.split('.')[0] != 'ES2002d':
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('.')[0])

    #
    #predict = []
    #
    #speaker_count = []

    ############# ОТРЕЗОК СИГНАЛА
    SEC = 0.5
    #############

    #path_to = 'AMI_amp4mean_without0_old_rttm' + v_vad + '_array1-01'  # withoun 0 - использование разметки csv с речевой активностью, old - секунды целочисленные, rttm0.0 - VAD Леши с порогом 0.0 (если нет - идеальная разметка)
    path_to = 'AMI_amp4mean_array1-01'

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
    
    for k in range(len(path_f)): # 169 range(70, 80) len(path_f)

    # CUSTOM CODE
        t_file = time.time()
    # compute audio
        audio, rate = sf.read(path_f[k], always_2d=True)
    #audio, rate = sf.read(args.audio, always_2d=True)

        #true_labels = get_labels_AMI('outputs\\AMIcsv\\' + name_f[k] + '.count.csv', SEC)
        # Загрузить фалик noSilence

    # downmix to mono
        audio = np.mean(audio, axis=1)
        
        length = (len(audio)//rate)//SEC
        #print(length)
        seconds = [i*SEC for i in range(0, int(length))]
        #print(seconds)
        # seconds - набор секунд, с которых (!) нужно обработать


        # НАПИСАТЬ ФУНКЦИЮ ПО УСИЛЕНИЮ ЗВУКА (ДЛЯ ВСЕЙ ЗАПИСИ)
        AMP = AMP_audio(audio)

        #seconds = get_silence(name_f[k], SEC)

        #print(audio.shape, rate)
        #print(audio.shape)
        #print(seconds)
        #print(int(seconds[0]*rate-1000), int((seconds[0]+SEC)*rate+1000))

        ##### Партия
        part = 1000 # 500
        sec_parts = len(seconds) // part
        if len(seconds) % part > 0:
            sec_parts += 1

        
        estimate = []
        for i in range(sec_parts):
            estimate.extend(count_part(audio, rate, seconds[i*part:(i+1)*part], model, scaler, SEC, AMP))
        
        '''
        t_stop = time.time()
        print("File: " + str(t_stop - t_file))
        t_file = time.time()
        
        estimate = []
        for i in range(len(seconds)):
            sec_begin = int(seconds[i]*rate)
            estimate.append(count(audio[sec_begin:sec_begin+SEC*rate], model, scaler, SEC, AMP)) 
        '''


        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name_f[k] + '.output.pickle', 'wb') as f:
            pickle.dump(estimate, f)

        #speaker_count.extend(estimate)
        print(name_f[k])
        t_stop = time.time()
        print("Time: " + str(t_stop - t_start))
        print("File: " + str(t_stop - t_file))

        if k % 10 == 0:
            print('Обработано: ' + str(k) + ' из ' + str(len(path_f)))



    print("Обработано: ", len(path_f))
    
    
    ################
    # Метрики
    ################
    
    names_audio = os.listdir('outputs\\AMIcsv') ########## ВЗЯТЬ ИМЕНА АУЙДИОФАЛОВ - сделано

    t_metrics = time.time()

    f1 = []

    true_l = []
    pred_l = []

    #for name_au in names_audio:
    for name in name_f: # [:10] [70:80]

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI(name, SEC, noSilence = False)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)
        
        # ТОЛЬКО ДЛЯ МОДЕЛЕЙ 1-4
        
        pred_labels = []
        for i in range(len(pr_labels)):
            #pred_labels.append(pr_labels[i] + 1)
            pred_labels.append(pr_labels[i])
        #print(len(true_labels), len(pred_labels)

                
        true = [0]*max(len(pred_labels), len(true_labels))
        pred = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred[:len(pred_labels)] = pred_labels[:]
        if len(true_labels) > len(pred_labels):
            print(name, len(true_labels), len(pred_labels))

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
        #print(true.shape, pred.shape, name)
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
    t_metrics = time.time()
'''
    f1 = []

    true_l = []
    pred_l = []

    #for name_au in names_audio:
    for name in name_f: # [:10] [70:80]

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI(name, SEC, noSilence = False)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)
        
        seconds = get_silence(name, SEC)
        pred_labels = []
        for i in range(seconds[-1] + 1):
            if i in seconds:
                #pred_labels.append(pr_labels[seconds.index(i)] + 1)
                pred_labels.append(pr_labels[seconds.index(i)])
            else:
                pred_labels.append(0)


        #print(len(true_labels), len(pred_labels))

        true = [0]*max(len(pred_labels), len(true_labels))
        pred = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred[:len(pred_labels)] = pred_labels[:]

        ### Классификация для разного количества классов
        
        num_cl = 2 # до n+
    
        for i in range(len(true)):
            if true[i] > num_cl:
                true[i] = num_cl
            if pred[i] > num_cl:
                pred[i] = num_cl
        
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
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_precision_allClasses.png')

    cm = metrics.confusion_matrix(true, pred, normalize='true') # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig('outputs\\' + args.model + '\\' + path_to + '\\_conf_matrix_recall_allClasses.png')

    with open('outputs\\' + args.model + '\\' + path_to + '\\_report_allClasses.txt', 'w') as f:
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
'''



