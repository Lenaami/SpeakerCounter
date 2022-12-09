# coding=utf-8

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow import keras

import soundfile as sf
import librosa
import pickle
import pandas as pd

import sklearn

import os

from utils_.extract_features import extract_features
from utils_.data_list import get_data_list

input_features = {
    'stft': 201,
    'melsp': 128,
    'mfcc': 40
}

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, audio_list, batch_size, audio_path, audio_type, n_classes, sec, features,
                 to_fit=True, dim=(500, 201), n_channels=1, shuffle=False, **kwargs):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """

        # считать файл разметки, узнать длину (последнее значение + dur)
        # разбить по sec и записать в listIDs имя файла и секунды для считывания, в labels кол-во        
        list_IDs = []
        labels = []
        k = 0

        # Завернуть в get_metrics
        # Создание list_IDs
        for audio_name in audio_list:
            file_csv = './data/AMI/Count_new/' + audio_name + '.count.csv' # путь к файлам

            DATA = pd.read_csv(file_csv, delimiter=',') # колонки - start, duration, count или (start, end, duration, count)
            data = DATA.values
            data[:, 1] = data[:, 0] + data[:, 1] # переделываю в start, end, count

            # разметка по секундам целого файла
            for i in range(int(data[-1, 1] // sec + 1)):
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
        #print(audio_list)

        self.list_IDs = list_IDs
        self.labels = labels
        self.audio = audio
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (sec*100, input_features[features]) #dim 
        self.n_channels = n_channels # (1 - для слоев CONV)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.features = features
        self.sec = sec
        self.eps = np.finfo(np.float).eps

        self.cur_audio_name = ''
        self.cur_audio = []
        self.cur_audio_rate = 0

        self.scaler = sklearn.preprocessing.StandardScaler()
        with np.load(os.path.join("models", 'scaler_' + self.features + '.npz')) as data:
            self.scaler.mean_ = data['arr_0']
            self.scaler.scale_ = data['arr_1']


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        '''
        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X
        '''

        ############ другой вариант
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    '''
    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y
    '''

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, self.n_channels, *self.dim))
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # меняем размерность на NHWC
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # переделать хард код

            if self.cur_audio_name != ID[1]:
                self.cur_audio_name = ID[1]
                self.cur_audio, self.cur_audio_rate = sf.read(self.audio[ID[1]], always_2d=True) # ID[1] - имя файла
                self.cur_audio = np.mean(self.cur_audio, axis=1)
                ### Для предобработанных файлов
                #with open('F:\\amicorpus_2\\original_pickle\\' + ID[1] + '.pickle', 'rb') as f:
                #    self.cur_audio = pickle.load(f)

            ### Для предобработанных файлов
            #sec_begin = int(ID[2]//self.sec)
            #X_1 = self.cur_audio[sec_begin]
            
            # Для обработки на ходу
            sec_begin = int(ID[2]*self.cur_audio_rate) # ID[2] - секунда, с которой нужно считать            
            audio_part = self.cur_audio[sec_begin:sec_begin+int(self.sec*self.cur_audio_rate)]

            X_0 = extract_features(audio_part, self.cur_audio_rate, self.features)                 

            # cut to input shape length (500 frames x 201 STFT bins)
            X_0 = self.scaler.transform(X_0)
            X_0 = X_0[:int(100*self.sec), :]
            X_1 = np.zeros(self.dim) 
            X_1[:X_0.shape[0], :] = X_0 

            # apply l2 normalization
            Theta = np.linalg.norm(X_1, axis=1) + self.eps
            X_1 /= np.mean(Theta)

            X_1 = np.expand_dims(X_1, axis=2) # меняем размерность на NHWC

            # Store sample
            X[i,] = X_1

            # Store class
            y[i] = self.labels[ID[0]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

