# coding=utf-8

import numpy as np
import soundfile as sf
import pandas as pd
import sklearn
import os

from torch.utils.data import Dataset
import torchaudio

from utils_.extract_features import extract_features
from utils_.data_list import get_data_list

input_features = {
    'stft': 201,
    'melsp': 128,
    'mfcc': 40
}

class AmiDataset(Dataset):
    def __init__(self, audio_list, audio_path, audio_type, n_classes, sec, features, marks_path,
                 n_channels=1, **kwargs):
        # считать файл разметки, узнать длину (последнее значение + dur)
        # разбить по sec и записать в listIDs имя файла и секунды для считывания, в labels кол-во
        list_IDs = []
        labels = []
        k = 0

        # Завернуть в get_metrics
        # Создание list_IDs
        for audio_name in audio_list:
            file_csv = marks_path + audio_name + '.count.csv' # путь к файлам

            DATA = pd.read_csv(file_csv, delimiter=',') # колонки - start, duration, count или (start, end, duration, count)
            data = DATA.values
            data[:, 1] = data[:, 0] + data[:, 1] # переделываю в start, end, count

            # разметка по секундам целого файла
            for i in range(int(data[-1, 1] // sec + (data[-1, 1] % sec > 16000 * 0.2))):
                persons = [0]
                for st, ed, pers in data:
                    if ((i+1)*sec - st) > 0 and (ed - i*sec) > 0:
                        persons.append(pers)
                max_pers = max(persons) # делаем для задачи 0-4+ диктора
                if max_pers > 4:
                    max_pers = 4

                list_IDs.append([k, audio_name, i*sec])
                k += 1
                labels.append(max_pers)

        # Создание словаря с путями до файлов
        audio = get_data_list(audio_list, audio_type, audio_path, use_dict=True)
        #print(audio)

        self.list_IDs = list_IDs
        self.labels = labels
        self.audio = audio
        self.dim = (int(sec*100), input_features[features]) #dim
        self.n_channels = n_channels # (1 - для слоев CONV)
        self.n_classes = n_classes

        self.features = features
        self.sec = sec
        self.eps = np.finfo(np.float).eps

        audios = {}
        for au in audio_list:
            wav, rate = sf.read(audio[au], always_2d=True)
            wav = np.mean(wav, axis=1)
            audios[au] = [wav, rate]

        self.audios = audios
        self.cur_audio_name = ''
        self.cur_audio = []
        self.cur_audio_rate = 0

        self.scaler = sklearn.preprocessing.StandardScaler()
        with np.load(os.path.join("./models", 'scaler_' + self.features + '_AMI.npz')) as data:
            self.scaler.mean_ = data['arr_0']
            self.scaler.scale_ = data['arr_1']

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):

        X = np.empty((self.n_channels, *self.dim))
        y = np.empty((1), dtype=int)

        # Generate data
        ID = self.list_IDs[idx]

        '''
        if self.cur_audio_name != ID[1]:
            self.cur_audio_name = ID[1]
            self.cur_audio, self.cur_audio_rate = sf.read(self.audio[ID[1]], always_2d=True) # ID[1] - имя файла
            self.cur_audio = np.mean(self.cur_audio, axis=1)
        
        sec_begin = int(ID[2]*self.cur_audio_rate) # ID[2] - секунда, с которой нужно считать
        audio_part = self.cur_audio[sec_begin:sec_begin+int(self.sec*self.cur_audio_rate)]
        '''

        sec_begin = int(ID[2] * self.audios[ID[1]][1]) # ID[2] - секунда, с которой нужно считать
        audio_part = self.audios[ID[1]][0][sec_begin:sec_begin+int(self.sec*self.audios[ID[1]][1])]

        try:
            X_0 = extract_features(audio_part, self.cur_audio_rate, self.features)
        except Exception:
            print(Exception)
            print(ID[1], ID[2], sec_begin, audio_part.shape)

            X_0 = np.empty(self.dim)


        # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = self.scaler.transform(X_0)
        X_0 = X_0[:int(100*self.sec), :]
        X_1 = np.zeros(self.dim)
        X_1[:X_0.shape[0], :] = X_0

        # apply l2 normalization
        Theta = np.linalg.norm(X_1, axis=1) + self.eps
        X_1 /= np.mean(Theta)

        #X_1 = np.expand_dims(X_1, axis=0) # меняем размерность на NHWC

        # Store sample
        X[0, :] = X_1

        # Store class
        y[:] = self.labels[ID[0]]

        return X, y