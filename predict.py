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

from tqdm import tqdm

from utils_.data import get_list_of_data
from utils_.count import count_audio
from utils_.model import load_model, create_model_name
from utils_.metrics import count_metrics

# функции (для подгрузки):
# - count_part (+ усиление звука)
# - count_part_probs (добавить для EER)
# - get_labels (возможно вызывать в metric)
# - metrics (с разными параметрами: модели с 0 и без, кол-во классов (группы); по кодовому слову (подается в функцию) определять тип метрики (или определять с помощью параметров))
# - get_rttm (секунды с участками речи) (предусмотреть vad и идеальную разметку)
# - для вывода результатов
# - составление списка файлов и имен для обработки
# - ^ раскидать все функции по папкам

# отдельные функции:
# - создание датасета
# - обучение (+ дообучение) (??? точно ли функция или отдельный файл)
# 

def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


def create_predict_name(audio_amp, vad, v_vad, inverse_vad, ms, add_name, **kwargs):

    '''
    Расшифровка имени:
    AMI - датасет и файлы для тестирования
    amp - метод усиления сигнала (mean/max) и коэффициент
    without0/1 - использование vad для участков речи/тишины
    old - разметка по целым секундам без смещений на милисекунды
    rttm - использование vad Леши и порог склейки
    '''

    name = 'AMI'
    if vad == 'test':
        name += '_test'
    name += '_array1-01_amp' + audio_amp
    if vad != 'none':
        if vad != 'test':                
            if inverse_vad:
                name += '_without1'
            else:
                name += '_without0'
        if ms == False:
            name += '_old'
        if vad == 'vad':
            name += '_rttm' + v_vad

    name += add_name

    return name




def predict(sec, features):

    v_vad ='2.0'

    # для настройки: vad, sec, n_classes, features

    args = {     
        'path_audio': 'F:\\amicorpus_1',
        'path_to_file_count': 'data\\AMI\\Count_new\\', # путь к разметке кол-ва дикторов
        'path_to_file_vad': 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_' + v_vad + '\\', # vad
        'path_to_file_ideal_vad': 'data\\AMI\\NoSilence_new\\', 
        'path_to_file_test': 'data\\AMI\\Test_new\\',

        'model_arch': 'LSTM', # 'LSTM', 'CNN'
        'dataset': 'AMI', # 'AMI', '               
        'sec': sec,
        'n_classes': 5, # 4 или 5
        'features': features, # 'stft', 'melsp', 'melsp_1', 'mfcc'

        'audio_amp': '4mean',
        'vad': 'test', # vad, ideal_vad, none, test (!)
        'v_vad': v_vad,
        'ms' : True, # разметка по милисекундам или целым секундам
        'inverse_vad': False, # определение речи или тишины

        'add_name': '' # дополнительное имя для теста

        #'path_to': 'AMI_amp4mean_without0_old_rttm' + v_vad + '_array1-01' 
    }

    # задать список параметров, которые можно менять (длительность, модель, классы, предобработка, версия vad (номер/идеальный))
    
    # Модель CountNet
    '''
    model = keras.models.load_model(
        os.path.join('C:\\Users\\Лена\\PythonCodeProjects\\CountNet-master\\models', 'CRNN.h5'),
        custom_objects={
            'class_mae': class_mae,
            'exp': K.exp
        }
    )
    '''
    model = load_model(**args)
    

    # только для stft (!)
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler_' + args['features'] + '.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    audio_names = get_list_of_data(**args)

    path_to = 'F:\\nirma\\SpeakerCounter-outputs\\' + create_model_name(**args) + '\\' + create_predict_name(**args) + '\\'
    #path_to = 'outputs\\CountNet_CRNN\\' + create_predict_name(**args) + '\\'

    if not os.path.exists(path_to + 'pkl'):
        os.makedirs(path_to + 'pkl')

    for name, path in tqdm(audio_names.items()):
        
        count_audio(name, path, model, scaler, path_to, **args) # подсчитывает дикторов на записи и сохраняет результат

    
    #count_audio('TS3003c', audio_names['TS3003c'], model, scaler, path_to, **args)

    count_metrics(audio_names.keys(), path_to, **args)
    #count_metrics(audio_names.keys(), path_to, only_predicted=False, **args)
    


    #print("Обработано: ", len(path_f))
    # код для предсказывания и записи

    # !!!!! название папки для записи составляется автоматически из параметров

    # код для метрик 
    # сделать EER (!)

    # результаты экспериментов добавить в отдельный файл csv (?)




    # сделать такой же код для train + дообучение (попробовать подгрузить модель, скомпилировать и запустить дообучение)
    # сделать код для графиков (*)
    # функция для создания датасета (разные длительности и разная предобработка)

def main():
    #sec = [5, 4, 3, 2, 1.5, 1, 0.5]
    #sec = [2, 1.5, 1, 0.5]
    sec = [1]
    for s in sec:
        predict(s, 'stft')
        #predict(s, 'melsp_1')
        #predict(s, 'mfcc')



if __name__ == '__main__':
    main()
