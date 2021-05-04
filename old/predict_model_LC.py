#############
# Команда для запуска
# python predict_model_LC.py F:/LibriCount10-0dB/test/audio --model 1

# НУЖНО МЕНЯТЬ РАЗМЕР ВХОДНЫХ ДАННЫХ
#############

import numpy as np
import soundfile as sf
import argparse
import os
import keras
import sklearn
import librosa
from keras import backend as K

import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

eps = np.finfo(np.float).eps


def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


def count(audio, model, scaler):
    # compute STFT
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    # apply global (featurewise) standardization to mean1, var0
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:500, :]

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]


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
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('.')[0])

    pred = []
    true = []

    for k in range(len(path_f)):

    # CUSTOM CODE

    # compute audio
        audio, rate = sf.read(path_f[k], always_2d=True)
    #audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
        audio = np.mean(audio, axis=1)
        estimate = count(audio, model, scaler)

        #
        pred.append(estimate)
        true.append(int(name_f[k].split('_')[0]))
        #

        if k % 200 == 0:
            print('Обработано: ' + str(k) + ' из ' + str(len(path_f)))


    print("Обработано: ", len(path_f))

    ################
    # Метрики
    ################

    print('Объединенные результаты по датасету')
    print(f1_score(true, pred, average='weighted'))

    print(metrics.classification_report(true, pred))

    print('Confusion matrix (Precision)')
    print(metrics.confusion_matrix(true, pred, normalize='pred'))  # 'true' - recall, 'pred' - precisoin, 'all'
    print('Confusion matrix (Recall)')
    print(metrics.confusion_matrix(true, pred, normalize='true'))

    with open('outputs\\' + args.model + '\\_report_LC.txt', 'w') as f:
        f.write('Файлы: ' + args.audio + '\n')
        f.write('Модель: ' + args.model + '\n')
        f.write('\n')
        f.write('F1-score\n')
        f.write('Объединенные результаты по датасету: ' + str(f1_score(true, pred, average='weighted')) + '\n')
        f.write('\n')
        f.write(str(metrics.classification_report(true, pred)) + '\n')
        f.write('\n')
        f.write('Confusion matrix (Precision)\n')
        f.write(str(metrics.confusion_matrix(true, pred, normalize='pred')) + '\n')
        f.write('Confusion matrix (Recall)\n')
        f.write(str(metrics.confusion_matrix(true, pred, normalize='true'))+ '\n')






