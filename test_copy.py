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
    #rand_labels = get_labels_AMI('outputs\\AMIcsv\\' + name_f[0] + '.count.csv', SEC)
    
    #print(rand_files)
    #print(len(rand_labels))
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
    #min_A = [[], [], [], [], []]
    #max_A = [[], [], [], [], []]
    #print(min_A)

    #for k in range(len(rand_files)):

    # CUSTOM CODE

    # compute audio
    audio, rate = sf.read(rand_files[0], always_2d=True)
    #audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
    audio = np.mean(audio, axis=1)
        #seconds = (len(audio)//rate)//SEC
        #print(seconds)
        #i = 0 
        #for i in range(seconds):
    i = 20
    au = audio[i*rate*SEC:(i+1)*rate*SEC]

    # compute STFT
    X = np.abs(librosa.stft(au, n_fft=400, hop_length=160)).T

    X *= 4

    #X = librosa.feature.melspectrogram(y=au, sr=rate, S=X.T, n_fft=400, hop_length=160).T # S = X.T

    #X = librosa.feature.mfcc(y=au, sr=rate, S=X.T, n_mfcc=40).T

    #X = librosa.power_to_db(X, ref=np.max)

    print(X.shape)
    

    # apply global (featurewise) standardization to mean1, var0
    #X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    #X = X[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!


    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    print(X, np.max(X), np.min(X))

    plot_spec(X, 'stft')
