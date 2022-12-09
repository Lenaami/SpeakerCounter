# coding=utf-8
import numpy as np
import soundfile as sf
import librosa
import pickle
import pandas as pd

import torch
from utils_.extract_features import extract_features

length = {}

def AMP_audio(audio, rate):

    seconds = len(audio)//rate 

    amp = 0
    X_max = []
    for i in range(seconds):
        X = np.abs(librosa.stft(audio[i*rate:(i+1)*rate], n_fft=400, hop_length=160)).T
        X_max.append(np.max(X))
        #if np.max(X) < 5 and AMP < np.max(X):
        #    AMP = np.max(X)

    if np.mean(X_max) < 4:
        amp = 4/(np.mean(X_max) + 0.001)
    else:
        amp = 1

    return amp


def count_part(audio, rate, seconds, model, scaler, sec, amp, features, **kwargs):

    eps = np.finfo(np.float).eps

    X_part = []

    for i in range(len(seconds)):
        sec_begin = int(seconds[i]*rate)
        audio_part = audio[sec_begin:sec_begin+int(sec*rate)]

        X_0 = extract_features(audio_part, rate, features)
        #X_0 *= amp

        # apply global (featurewise) standardization to mean1, var0
        X_0 = scaler.transform(X_0)

        # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = X_0[:int(100*sec), :]
        X = np.zeros((int(100*sec), X_0.shape[1]))
        X[:X_0.shape[0], :] = X_0 

        # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        X_part.append(X)

    X_part = np.asarray(X_part)

    # add sample dimension
    #if len(model.input_shape) == 4:
    X_part = X_part[:, np.newaxis, ...] #X = X[np.newaxis, ...]

    #ys = model.predict(X_part, verbose=0) # Keras  #return np.argmax(ys, axis=1)

    #X_part = torch.tensor(X_part).cuda().float()
    X_part = torch.tensor(X_part).float()#.cuda()
    ys = model.inference(X_part)
    ys = ys.detach().cpu().numpy()#.argmax(axis=1)
    return ys

def count_part_n_sec(audio, rate, seconds, model, scaler, sec, amp, features, **kwargs):

    eps = np.finfo(np.float).eps
    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    X_part = []

    #print(seconds)

    # Определяем кол-во частей, на которые делится запись
    n_sec = kwargs['n_sec']
    #n_sec = 1.5
    n_parts = int(sec // n_sec) + (sec % n_sec > 0)
    #print(n_parts)

    for i in range(len(seconds)):
        sec_begin = int(seconds[i]*rate)
        audio_part = audio[sec_begin:sec_begin+int(sec*rate)]


        # другие признаки
        #print(audio_part.shape, '\t', sec_begin, sec_begin+int(sec*rate))

        X_0 = np.abs(librosa.stft(audio_part, n_fft=400, hop_length=160)).T
        #X_0 *= amp

        if features == 'stft':

            #X_0 *= amp
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 201))

        if features == 'melsp_1':

            X_0 = librosa.feature.melspectrogram(y=audio_part, sr=rate, S=X_0.T, n_fft=400, hop_length=160).T
            X_0 = librosa.power_to_db(X_0, ref=np.max)
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 128))

        if features == 'mfcc':

            X_0 = librosa.feature.mfcc(y=audio_part, sr=rate, S=X_0.T, n_mfcc=40).T
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 40))

        #######

        # apply global (featurewise) standardization to mean1, var0
        #X_0 = scaler.transform(X_0)

        # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = X_0[:int(100*sec), :]
        X[:X_0.shape[0], :] = X_0

        # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        #Делим полученные признаки на несколько частей
        #print(X.shape)
        for i in range(n_parts):
            if i == n_parts - 1: #последний элемент
                X_part.append(X[-int(100*n_sec):, :])
                #print(X[-int(100*n_sec):, :].shape)
            else:
                X_part.append(X[i*int(100*n_sec):(i+1)*int(100*n_sec), :])
                #print(i*int(100*n_sec), (i+1)*int(100*n_sec))

        #X_part.append(X)

    # add sample dimension
    #X = X[np.newaxis, ...]

    X_part = np.asarray(X_part)

    X_part = X_part[:, np.newaxis, ...]
    #if len(model.input_shape) == 4:
    #    X_part = X_part[:, np.newaxis, ...]

    X_part = torch.tensor(X_part).float()
    ys = model.inference(X_part)
    ys = ys.detach().cpu().numpy()#.argmax(axis=1)

    #Выделить лучшие строки (probs не сохраняем)
    ys = np.argmax(ys, axis=1)

    yys = []
    for i in range(ys.shape[0] // n_parts):
        yys.append(max(ys[i*n_parts:(i+1)*n_parts]))

    return yys





def get_seconds_VAD(name, vad, sec, inverse_vad, ms, path_to_file_vad, path_to_file_ideal_vad, path_to_file_test, **kwargs): # inverse - взять те промежутки, где VAD НЕ отметил речь

    if vad == 'vad':
        # Использование VADа
        #file = path_to_file_vad + name + '.segments.rttm'
        file = path_to_file_vad + name + '.rttm'

        data = []
        with open(file, encoding='utf-8') as r_file:
            for line in r_file:
                ln = line.split(' ')
                #data.append([float(ln[2]), float(ln[3])])
                data.append([float(ln[3]), float(ln[4])])

    else:
        # Идеальная разметка
        if vad == 'ideal_vad':
            file = path_to_file_ideal_vad + name + '.no_silence.csv' # путь к файлам

        # Тестовая разметка
        if vad == 'test':
            file = path_to_file_test + name + '.test_count.csv'

        DATA = pd.read_csv(file, delimiter=',') # колонки - start, duration
        data = DATA.values

   
    seconds = []

    # с начала (которое указано), по одной секунде, оставшийся отрезок меньше 0.1-0.2?, -> нужные секунды
    # пример 11.09, 4.4399999999999995 -> 11.09, 12.09, 13.09, 14.09, 15.09 --> далее перечисление продолжается со следующего промежутка

    if ms:
        # Разметка по миллисекундам
        for st, dur in data:
            times = dur // sec + (dur % sec > float(sec) / 5) # длительность последнего фрагмента должна превышать 0,2 от секунд обработки
        
            for i in range(int(times)):
                seconds.append(st + i*sec)
   
    else:
        # Сравнение с общей разметкой по 1 секунде
        for st, dur in data:
            times = (dur + (st - int(st))) // sec + ((dur + (st - int(st))) % sec > 0) # вычисление кол-ва секунд, которые нужно обработать

            #print(times)

            for i in range(int(times)): # int(times) ?????
                seconds.append(int(st) + i*sec)
                #print(int(st) + i*sec)

    seconds = sorted(list(set(seconds))) # убираем повторения (если есть)

    if inverse_vad == True:        
        seconds_all = list(np.arange(length[name])) # список секунд (нужно знать длину файлов)  ### НЕ ARANGE А, I*SEC
        return [s for s in seconds_all if s not in seconds] # not in - участки без речи, in участки с речью

    return seconds  


def count_audio(name, path, model, scaler, path_to, vad, sec, **kwargs):

    # взять файл и предсказать значения для него
    # использование vad, rttm

    #t_file = time.time()

    # compute audio
    audio, rate = sf.read(path, always_2d=True)

    #length[name] = (len(audio)//rate)//sec # нужно для inverse vad
        
    # downmix to mono
    audio = np.mean(audio, axis=1)

    #amp = AMP_audio(audio, rate)
    amp = 1
        
    if vad != 'none':
        seconds = get_seconds_VAD(name, vad, sec, **kwargs) # использование vad
    else:
        length = int((len(audio)//rate)//sec) + ((len(audio)//rate) % sec > 0) # не учитывается остаток отрезка (если есть)  - делим аудио на отрезки (это их количество)
        seconds = [i*sec for i in range(0, length)] # все секунды подаем на вход

    ##### Партия
    part = int(2000 // sec) # длина партии
    sec_parts = len(seconds) // part + (len(seconds) % part > 0) # количество партий

    estimate = []
    for i in range(sec_parts):
        estimate.extend(count_part(audio, rate, seconds[i*part:(i+1)*part], model, scaler, sec, amp, **kwargs))
        #estimate.extend(count_part_n_sec(audio, rate, seconds[i*part:(i+1)*part], model, scaler, sec, amp, **kwargs))

    # Не сохраняем pickle для sizeCN
    with open(path_to + 'pkl/probs/' + name + '.output_probs.pickle', 'wb') as f:
        pickle.dump(estimate, f)

    estimate = np.asarray(estimate)
    if estimate.shape[0] == 0:
        estimate == []
    else:
        #estimate[:, 0] = 0 # только для vad
        estimate = np.argmax(estimate, axis=1)



    with open(path_to + 'pkl/' + name + '.output.pickle', 'wb') as f:
        pickle.dump(estimate, f)
    

    #t_stop = time.time()
    #print("File: " + str(t_stop - t_file))
