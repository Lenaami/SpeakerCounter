# coding=utf-8

import os
import yaml

def get_data_list(names, audio_type, audio_path, use_dict=False, **kwargs):

    '''Функция для получения списка используемых файлов
    Возвращает список названий файлов и пути

    names (list)     - список имен аудио
    audio_type (str) - тип аудио ('Array1-01' / 'Mix-Headset')
    audio_path (str) - путь к данным
    '''

    audio_names = []
    audio_paths = []

    for d, dirs, files in os.walk(audio_path):
        for f in files:
            path = os.path.join(d, f) # формирование полного адреса
            name = f.split('.')
            if name[0] in names and name[1] == audio_type and name[2] == 'wav':
                audio_names.append(name[0]) # добавление имени в список
                audio_paths.append(path) # добавление адреса в список

    if use_dict:
        audio = {}
        for i in range(len(audio_names)):
            audio[audio_names[i]] = audio_paths[i]        
        return audio
    else:
        return audio_names, audio_paths

def get_data_names(set_path, path_to_file_count, vad, path_to_file_vad, path_to_file_ideal_vad, path_to_file_test, set_ami='all', **kwargs):

    """Функция для получения списка используемых имен файлов
    Возвращает список названий файлов
    set (str): 'all', 'train', 'test', 'val'
    """

    # Берем список нужных имен
    if set_ami == 'all':  # использование всех записей
        files = os.listdir(path_to_file_count)
        names = [f.split('.')[0] for f in files]
    elif set_ami == 'alex':
        names = ['EN2006b', 'ES2004d', 'IB4003', 'IN1005', 'IS1000a', 'TS3004b', 'EN2001e', 'ES2005b', 'IN1009', 'IS1004c', 'TS3012a', 'TS3009b', 'EN2002d', 'ES2005c', 'TS3012d']
    else:
        with open(set_path) as f:
            ami_sets = yaml.load(f, Loader=yaml.FullLoader)
        names = ami_sets[set_ami]

    # Список нужных имен из разметок
    if vad != 'none':  # для 'vad', 'ideal_vad', 'test'
        if vad == 'vad':
            files = os.listdir(path_to_file_vad)
        elif vad == 'ideal_vad':
            files = os.listdir(path_to_file_ideal_vad)
        elif vad == 'test':
            files = os.listdir(path_to_file_test)

        names_vad = [f.split('.')[0] for f in files]
    
        names = list(set(names) & set(names_vad))  # пересечение множеств

    return names