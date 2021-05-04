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

path = 'outputs\\AMIwords\\'

fnames_cur = os.listdir(path)
print('Количество файлов в папке:', len(fnames_cur))

######## Список файлов аудио
'''
files_audio = []
for file in fnames_cur:
    files_audio.append(file.split('.')[0])

files_audio = list(set(files_audio))
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

        for dt in root.findall('w'):
            if dt.get('punc') is None and dt.get('starttime') is not None:
                if data != [] and data[-1][1] == float(dt.get('starttime')):
                    data[-1][1] = float(dt.get('endtime'))
                else:    
                    data.append([float(dt.get('starttime')), float(dt.get('endtime')), person])
    
    data_sort = sorted(data, key=lambda x: x[0])
    DATA = pd.DataFrame(data_sort, columns=['start', 'end', 'person'])

    time = list(DATA['start'].values)
    time.extend(list(DATA['end'].values))
    ###
    time.append(0.0)
    time.append(length[file_au])
    ###

    time = sorted(list(set(time)))
    ###
    time = time[:time.index(length[file_au])+1]
    ###


    speakers = ['']*(len(time) - 1)
    #interrupt = []
    count = []
    #no_silence = []

    for i in range(len(time) - 1):
        for st, ed, pers in data_sort:
            if (time[i+1] - st) > 0 and (ed - time[i]) > 0:
                speakers[i] += pers
        speakers[i] = len(speakers[i])
        #if speakers[i] > 1:
        #    interrupt.append([time[i], time[i+1], time[i+1] - time[i], speakers[i]])
        #count.append([time[i], time[i+1], time[i+1] - time[i], speakers[i]])
        #count.append([time[i], time[i+1] - time[i], speakers[i]])
        count.append([time[i], time[i+1], speakers[i]])
        #if speakers[i] > 0:
        #    no_silence.append([time[i], time[i+1]])   


    count_1 = []
    sec = 1

    for i in range(int(count[-1][1] // sec + 1)):
        persons = [0]
        for st, ed, pers in count:
            if ((i+1)*sec - st) > 0 and (ed - i*sec) > 0:
                persons.append(pers)
        count_1.append([i*sec, sec, max(persons)])


    # Промежуки с речью
    '''
    ns = []
    for i in range(len(no_silence) - 1):
        if ns != [] and ns[-1][1] == no_silence[i][0]:
            ns[-1][1] = no_silence[i][1]
        else:
            ns.append([no_silence[i][0], no_silence[i][1]])

    for i in range(len(ns)):
        ns[i][1] = ns[i][1] - ns[i][0]
    '''
      
    #INTER = pd.DataFrame(interrupt, columns=['start', 'end', 'duration', 'count'])
    #INTER.to_csv(path_to + file_au + '.interrupt.csv', index=False)

    #INTER = pd.DataFrame(count, columns=['start', 'end', 'duration', 'count'])
    INTER = pd.DataFrame(count_1, columns=['start', 'duration', 'count'])
    INTER.to_csv('outputs\\Ideal\\AMIcsv_new_1sec\\' + file_au + '.count.csv', index=False)

    #INTER = pd.DataFrame(ns, columns=['start', 'duration'])
    #INTER.to_csv('outputs\\AMIsilence\\' + file_au + '.no_silence.csv', index=False)

    '''
    array = []

    #SPEECH EN2001a 2.91 0.03 формат

    for st, dur, sp in count_1:
        array.append('SPEECH ' + file_au + ' ' + str(st) + ' ' + str(dur) + ' ' + str(sp) + '\n')

    with open('outputs\\Ideal\\AMIrttm_1\\' + file_au + '.count.rttm', 'w') as f:
        f.writelines(array)
    '''

    #print(len(array))

