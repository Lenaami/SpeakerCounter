# coding=utf-8

import numpy as np
import librosa


def extract_features(audio, rate, features):

    ''' Функция для вычисления признаков аудио
    Возвращает признаки аудио
    '''
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    if features == 'melsp':
        X = librosa.feature.melspectrogram(y=audio, sr=rate, S=X.T, n_fft=400, hop_length=160).T
        #X = librosa.feature.melspectrogram(S=X.T, n_mels=128, sr=rate).T
        X = librosa.power_to_db(X, ref=np.max)

    if features == 'mfcc':
        X = librosa.feature.mfcc(y=audio, sr=rate, S=X.T, n_mfcc=40).T

    return X

