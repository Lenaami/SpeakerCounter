import numpy as np
import soundfile as sf
import os
import sklearn
import librosa
import pickle

eps = np.finfo(np.float).eps

def scaler_fit(path_f, labels_f, feature):

    DATA = []

    for k in range(len(path_f)):

    # compute audio
        audio, rate = sf.read(path_f[k], always_2d=True)

    # downmix to mono
        audio = np.mean(audio, axis=1)

        X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

        #if feature == 'stft':

        if feature == 'melsp':
            X = librosa.feature.melspectrogram(y=audio, sr=rate, S=X.T, n_fft=400, hop_length=160).T
            X = librosa.power_to_db(X, ref=np.max)

        if feature == 'mfcc':
            X = librosa.feature.mfcc(y=audio, sr=rate, S=X.T, n_mfcc=40).T

    # apply global (featurewise) standardization to mean1, var0
        #X = scaler.transform(X)


    # cut to input shape length (500 frames x 201 STFT bins)
        X = X[:500, :]

    # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        if labels_f[k] < 5 and labels_f[k] >= 0:

            DATA.extend(X)


    DATA = np.asarray(DATA)
    print(DATA.shape)


    scaler = sklearn.preprocessing.StandardScaler()

    scaler.fit(DATA)

    np.savez('models/scaler_' + feature, scaler.mean_, scaler.scale_)

    data = np.load('models/scaler_' + feature + '.npz')

    print('Mean\t', data['arr_0'].shape)
    print('Scale\t', data['arr_1'].shape)


    print('Scaler prepare complete! Feature: ' + feature)

if __name__ == '__main__':

    audio_path = 'F:/LibriCount10-0dB/test/audio'

    '''
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        #print('Scaler: mean - ', data['arr_0'], ' , scale - ', data['arr_0'])
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']
    '''

    path_f = []
    labels_f = []
    for d, dirs, files in os.walk(audio_path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            path_f.append(path) # добавление адреса в список
            labels_f.append(int(f.split('_')[0]))

    #features = ['stft', 'melsp', 'mfcc']
    features = ['melsp', 'mfcc']
    for feat in features:
        scaler_fit(path_f, labels_f, feat)
