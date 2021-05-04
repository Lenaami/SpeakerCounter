import numpy as np
#import matplotlib.pyplot as plt
#from scipy.io import wavfile
#import IPython
import pickle
import os.path
import xml.etree.ElementTree as ET 
import re
import pandas as pd

import soundfile as sf

#path = 'outputs\\AMIwords\\'
path = 'outputs\\AMIsegments\\'

fnames_cur = os.listdir(path)
print('Количество файлов в папке:', len(fnames_cur))

######## Список файлов аудио
'''
files_audio = []
for file in fnames_cur:
    files_audio.append(file.split('.')[0])
'''

path_f = []
files_audio = []
for d, dirs, files in os.walk('F:\\amicorpus_1'): # путь к файлам
    for f in files:
        path_1 = os.path.join(d,f) # формирование адреса
        #if f.split('.')[0] != 'ES2002d':
        path_f.append(path_1) # добавление адреса в список
        files_audio.append(f.split('.')[0])

files_audio = list(set(files_audio))

length = {}
a = 2
        
######## Подготовка csv для каждого аудио
for file_au in files_audio:

    audio, rate = sf.read('F:\\amicorpus_1\\' + file_au + '\\audio\\' + file_au + '.Array1-01.wav', always_2d=True)
    length[file_au] = len(audio)/float(rate)

    files = []
    for name in fnames_cur:
        if file_au in name:
            files.append(name)

    #print(file_au)

    data = []
    for file in files:

        #print(file)

        tree = ET.parse(path + file)
        root = tree.getroot()

        person = file.split('.')[1]

        #for dt in root.findall('w'):
        for dt in root.findall('segment'):
            #if dt.get('punc') is None and dt.get('starttime') is not None:
            #    if data != [] and data[-1][1] == float(dt.get('starttime')):
            #        data[-1][1] = float(dt.get('endtime'))
            #    else:    
            #        data.append([float(dt.get('starttime')), float(dt.get('endtime')), person])
            data.append([float(dt.get('transcriber_start')), float(dt.get('transcriber_end')), person])
    
    data_sort = sorted(data, key=lambda x: x[0])
    DATA = pd.DataFrame(data_sort, columns=['start', 'end', 'person'])

    time = list(DATA['start'].values)
    time.extend(list(DATA['end'].values))
    # добавление начала файла (0.0) и конца (length)
    time.append(0.0)
    time.append(length[file_au]) # Добавляем конец файла, чтобы заполнить 0-ми


    time = sorted(list(set(time)))

    #print('Длина файла ', length[file_au], 'Последний элемент ', time[-1])
    #if length[file_au] != time[-1]:
    #    print('Не совпадает. Файл: ' + file_au)

    #print('Индекс последнего ', time.index(length[file_au]), ' Длина массива ',  len(time))
    time = time[:time.index(length[file_au])+1] # Обрезаем по конец файла, чтобы не попало лишнее из разметки
    #print('Последний элемент', time[-1])

    #print('-------------------------')


    speakers = ['']*(len(time) - 1)
    #interrupt = []
    count = []
    no_silence = []

    for i in range(len(time) - 1):
        for st, ed, pers in data_sort:
            if (time[i+1] - st) > 0 and (ed - time[i]) > 0:
                speakers[i] += pers
        speakers[i] = len(speakers[i])
        #if speakers[i] > 1:
        #    interrupt.append([time[i], time[i+1], time[i+1] - time[i], speakers[i]])
        #count.append([time[i], time[i+1], time[i+1] - time[i], speakers[i]])
        if speakers[i] == 5:
            print(file_au, time[i], time[i+1])
        count.append([time[i], time[i+1] - time[i], speakers[i]])
        if speakers[i] > 0:
            no_silence.append([time[i], time[i+1]])   


    ########## Поиск подходящих примеров

    duration = [[], [], [], [], [], []]

    for st, dur, cnt in count:
        #print(int(cnt))
        duration[int(cnt)].append([st, dur])
        #print([len(duration[i]) for i in range(5)])

    count_1 = []
    
    ids = [4, 5, 3, 2, 1, 0]

    #for i in range(5, -1, -1):
    for i in ids:

        dur = pd.DataFrame(duration[i])
        if len(dur) > 0: dur = dur.sort_values(by=1, ascending=False) # сортировка по длительности
        dur = dur.values

        np.random.shuffle(dur) # перемешка (нужна ли сортировка?)
    
        len_cnt = 0
        if i == 4:
            len_max = 0
        for st, d in dur:
            if d > 0.2 and (len_cnt < len_max or i == 4):
                if (len_cnt + d) - len_max > 0 and i != 4:
                    minus = (len_cnt + d) - len_max
                    if d - minus > 0.2:
                        count_1.append([st, d - minus])
                else:
                    count_1.append([st, d])
                len_cnt += d
                if i == 4:
                    len_max += d

    count_2 = pd.DataFrame(count_1)
    if len(count_2) > 0: count_2 = count_2.sort_values(by=0)
    count_2 = count_2.values
    if len(count_2) == 0: count_2 = np.array([[0, 0]])

    ############

    # Промежуки с речью
    
    ns = []
    for i in range(len(no_silence) - 1):
        if ns != [] and ns[-1][1] == no_silence[i][0]:
            ns[-1][1] = no_silence[i][1]
        else:
            ns.append([no_silence[i][0], no_silence[i][1]])

    for i in range(len(ns)):
        ns[i][1] = ns[i][1] - ns[i][0]
    
      
    #INTER = pd.DataFrame(interrupt, columns=['start', 'end', 'duration', 'count'])
    #INTER.to_csv(path_to + file_au + '.interrupt.csv', index=False)

    #COUNT = pd.DataFrame(count, columns=['start', 'end', 'duration', 'count'])
    COUNT = pd.DataFrame(count, columns=['start', 'duration', 'count'])
    COUNT.to_csv('data\\AMI\\Count_new\\' + file_au + '.count.csv', index=False)

    TEST = pd.DataFrame(count_2, columns=['start', 'duration'])
    TEST.to_csv('data\\AMI\\Test_new\\' + file_au + '.test_count.csv', index=False)

    NOSILENCE = pd.DataFrame(ns, columns=['start', 'duration'])
    NOSILENCE.to_csv('data\\AMI\\NoSilence_new\\' + file_au + '.no_silence.csv', index=False)

    print('Done:\t', file_au)

    '''
    array = []

    #SPEECH EN2001a 2.91 0.03 формат

    for st, dur, sp in count:
        array.append('SPEECH ' + file_au + ' ' + str(st) + ' ' + str(dur) + ' ' + str(sp) + '\n')

    with open('outputs\\Ideal\\AMIrttm\\' + file_au + '.count.rttm', 'w') as f:
        f.writelines(array)

    '''

