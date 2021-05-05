import os
import yaml

def get_data_list(names, audio_type, audio_path, use_dict=False):

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
            if name[0] in names and name[1] == audio_type:
                audio_names.append(name[0]) # добавление имени в список
                audio_paths.append(path) # добавление адреса в список

    if use_dict:
        audio = {}
        for i in range(len(audio_names)):
            audio[audio_names[i]] = audio_paths[i]        
        return audio
    else:
        return audio_names, audio_paths

def get_data_list_names(**kwargs):

    '''Функция для получения списка используемых имен файлов
    Возвращает список названий файлов
    '''

    # Берем нужные переменные из словаря
    audio_path = kwargs['path_audio']
    vad = kwargs['vad']
    path_to_file_vad = kwargs['path_to_file_vad']
    path_to_file_ideal_vad = kwargs['path_to_file_ideal_vad']

    # взять список нужных имен (!)
    # если используются доп разметки (rttm vad'a) - взять и оттуда
    # создать пересечение этих множеств 
    # создать список имен и путей до нужных файлов (Array1-01/Mix-Headset)

    with open('yaml/ami_sets_mini.yaml') as f:
        ami_sets = yaml.load(f, Loader=yaml.FullLoader)

    if vad == 'vad':
        files = os.listdir(path_to_file_vad)
    else: # для 'ideal_vad', 'none'
        files = os.listdir(path_to_file_ideal_vad)
    
    names = [f.split('.')[0] for f in files]

    '''
    audio_names = {}

    for d, dirs, files in os.walk(audio_path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            name = f.split('.')[0]
            if name in names:
                audio_names[name] = path # добавление адреса в список
    '''

    return names