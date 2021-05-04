import numpy as np
import soundfile as sf
import argparse
import os
import sklearn
import librosa
import librosa.display

import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

eps = np.finfo(np.float).eps

def plot_spec(X, name):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X.T) #x_axis='fft', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    plt.tight_layout()
    plt.show()

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


    # save as svg file
    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    ####### LibriCount
    '''
    path = 'F:/LibriCount10-0dB/test/audio'

    path_f = []
    name_f = []
    for d, dirs, files in os.walk(path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('_')[0]) # labels

    indexes = np.arange(len(path_f))
    np.random.shuffle(indexes)
    
    rand_files = []
    rand_labels = []
    for idx in indexes[:]:
        rand_files.append(path_f[idx])
        rand_labels.append(name_f[idx])
    '''
    ########
    ######## Ami Corpus

    path = 'F:/AMIcorpus_1/EN2001a/audio' #IS1007c IN1014(good), EN2001a, ES2003a

    path_f = []
    name_f = []
    for d, dirs, files in os.walk(path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('.')[0])

    #indexes = np.arange(len(path_f))
    #np.random.shuffle(indexes)
    SEC = 1

    rand_files = [path_f[0]]
    #rand_files = ['F:/1amicorpus/loud/EN2001a.Mix-Headset.wav']
    rand_labels = get_labels_AMI('outputs\\AMIcsv\\' + name_f[0] + '.count.csv', SEC)
    
    #print(rand_files)
    print(len(rand_labels))
    ########

    ############# ОТРЕЗОК СИГНАЛА
    SEC = 1
    #############

    ###### !!!!!!!!!!!!! Унифицировать выход !!!!!!!!!!!

    #if not os.path.exists('outputs\\' + args.model):
    #    os.makedirs('outputs\\' + args.model)
    
    #true_labels = get_labels_AMI('outputs\\AMIcsv\\' + 'EN2001a' + '.count.csv', SEC)
    #print(true_labels)
    
    #min_A = [[], [], [], [], [], [], [], [], [], [], []]
    #max_A = [[], [], [], [], [], [], [], [], [], [], []]
    min_A = [[], [], [], [], []]
    max_A = [[], [], [], [], []]
    #print(min_A)

    for k in range(len(rand_files)):

    # CUSTOM CODE

    # compute audio
        audio, rate = sf.read(rand_files[k], always_2d=True)
    #audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
        audio = np.mean(audio, axis=1)
        seconds = (len(audio)//rate)//SEC
        print(seconds)
        #i = 0 
        for i in range(seconds):

            au = audio[i*rate*SEC:(i+1)*rate*SEC]

    # compute STFT
            X = np.abs(librosa.stft(au, n_fft=400, hop_length=160)).T

            if np.max(X) > 5:
                AMP = 1
            else:
                AMP = 5/(np.max(X) + 0.001) # Поиграть с этой величиной
            #print(np.mean(X), AMP)
            X *= AMP

    # apply global (featurewise) standardization to mean1, var0
        #print(X)
        #plot_spec(X, 'Спектр ' + rand_labels[k] + ' Min: ' + str(np.min(X)) + ' Max: ' + str(np.max(X)))
            speaker = 0
            if i < len(rand_labels):
                speaker = rand_labels[i]
        #print(rand_labels[k] + ' Min: ' + str(np.min(X)) + ' Max: ' + str(np.max(X)))
            min_A[speaker].append(np.min(X)) # заменить i на k (LibriCount)
            max_A[speaker].append(np.max(X))

            #min_A[int(rand_labels[i])].append(np.min(X)) # заменить i на k (LibriCount)
            #max_A[int(rand_labels[i])].append(np.max(X))


            X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
        #X = X[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!

        #plot_spec(X, 'Нормализация 1 ' + rand_labels[k] + ' Min: ' + str(np.min(X)) + ' Max: ' + str(np.max(X)))

    # apply l2 normalization
            Theta = np.linalg.norm(X, axis=1) + eps
            X /= np.mean(Theta)

        #plot_spec(X, 'Нормализация 2 ' + rand_labels[k] + ' Min: ' + str(np.min(X)) + ' Max: ' + str(np.max(X)))

    # проверить:
    # - посмотреть что вообще выводит stft с нормализацией и без 
        # - список файлов
        # - рандомно выбрать несколько штук, указать класс
        # - вывести графики спектров (librosa?) + min, max значения
        #
    # - сравнить спектры аудио из ami и lc
        # - выбрать запись ami
        # - считать разметку, выбрать рандомно моменты и указать классы
        # - вывести спектры; min, max значения
        #
    # - посмотреть на каком уровне в среднем амплитуды (!) спектров lc для разных классов
    # - так же для ami
    # - придумать функцию (умножение амплитуд на какой-то коэф), чтобы амплитуды были примерно равные










        #with open('outputs\\' + args.model + '\\AMI_loud_2\\' + name_f[k] + '.output.pickle', 'wb') as f:
        #    pickle.dump(estimate, f)

        #speaker_count.extend(estimate)

        #if k % 10 == 0:
        #    print('Обработано: ' + str(k) + ' из ' + str(len(path_f)))

    #with open('outputs\\output.pickle', 'wb') as f:
    #    pickle.dump(predict, f)
    for i in range(len(min_A)):

        print(i, 'Ex: ', len(min_A[i]))
        print('Mean min: ', np.mean(min_A[i]), '\tMin: ', min(min_A[i]), '\tMax: ', max(min_A[i]))
        print('Mean max: ', np.mean(max_A[i]), '\t\tMin: ', min(max_A[i]), '\tMax: ', max(max_A[i]))

    #print("Обработано: ", len(path_f))