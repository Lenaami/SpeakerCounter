#############
# Команда для запуска
# python predict_model.py F:/1amicorpus/loud --model 5
# python predict_model.py f:/amicorpus_1 --model 8
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

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

eps = np.finfo(np.float).eps


def count(audio, model, scaler, sec):

    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    ####### ТУПОЕ УСИЛЕНИЕ АМПЛИТУДЫ
    
    if np.max(X) > 5:
        AMP = 1
    else:
        AMP = 5/(np.max(X) + 0.001) # Поиграть с этой величиной / попробовать умножать на статичный коэф. (4?)
    X *= AMP
    
    #######

    # apply global (featurewise) standardization to mean1, var0
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]

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
            path_f.append(path) # добавление адреса в список
            name_f.append(f.split('.')[0])

    #
    #predict = []
    #
    #speaker_count = []

    ############# ОТРЕЗОК СИГНАЛА
    SEC = 1
    #############

    path_to = 'AMI_amp5_array1-01'

    ###### !!!!!!!!!!!!! Унифицировать выход !!!!!!!!!!! - вроде сделано (почти - номер модели)

    if not os.path.exists('outputs\\' + args.model):
        os.makedirs('outputs\\' + args.model)
    
    if not os.path.exists('outputs\\' + args.model + '\\' + path_to):
        os.makedirs('outputs\\' + args.model + '\\' + path_to)

    #true_labels = get_labels_AMI('outputs\\AMIcsv\\' + 'EN2001a' + '.count.csv', SEC)
    #print(true_labels)
    

    for k in range(len(path_f)): # range(70, 80) len(path_f)

    # CUSTOM CODE

    # compute audio
        audio, rate = sf.read(path_f[k], always_2d=True)
    #audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
        audio = np.mean(audio, axis=1)
        seconds = (len(audio)//rate)//SEC
        estimate = []
        for i in range(seconds):
            estimate.append(count(audio[i*rate*SEC:(i+1)*rate*SEC], model, scaler, SEC))
    #estimate = count(audio, model, scaler)

        #
        #predict.append([name_f[k].split('_')[0], estimate[0]])
        #

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name_f[k] + '.output.pickle', 'wb') as f:
            pickle.dump(estimate, f)

        #speaker_count.extend(estimate)
        print(name_f[k])

        if k % 10 == 0:
            print('Обработано: ' + str(k) + ' из ' + str(len(path_f)))

    #with open('outputs\\output.pickle', 'wb') as f:
    #    pickle.dump(predict, f)

    print("Обработано: ", len(path_f))
    #print("Speaker Count Estimate: ", estimate)
    

    ################
    # Метрики
    ################
    
    names_audio = os.listdir('outputs\\AMIcsv') ########## ВЗЯТЬ ИМЕНА АУЙДИОФАЛОВ - сделано

    f1 = []

    true_l = []
    pred_l = []

    #for name_au in names_audio:
    for name in name_f: # [:10] [70:80]

        #name = name_au.split('.')[0]

        #with open('outputs\\AMItrue\\' + name + '.pickle', 'rb') as f:
        #    true_labels = pickle.load(f)

        true_labels = get_labels_AMI('outputs\\AMIcsv\\' + name + '.count.csv', SEC)

        with open('outputs\\' + args.model + '\\' + path_to + '\\' + name + '.output.pickle', 'rb') as f:
            pred_labels = pickle.load(f)
        
        true = [0]*max(len(pred_labels), len(true_labels))
        pred = [0]*max(len(pred_labels), len(true_labels))
        true[:len(true_labels)] = true_labels[:]
        pred[:len(pred_labels)] = pred_labels[:]

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

#'''






