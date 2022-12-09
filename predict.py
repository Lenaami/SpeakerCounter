import numpy as np
import os
import sklearn

import yaml
from tqdm import tqdm

from utils_.data_list import get_data_names, get_data_list
from utils_.data import get_list_of_data
from utils_.count import count_audio
from utils_.model import load_model, create_model_name
from utils_.metrics import count_metrics


'''
def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )
'''

def create_predict_name(audio_amp, vad, v_vad, inverse_vad, ms, add_name_test, **kwargs):

    """
    Расшифровка имени:
    AMI - датасет и файлы для тестирования
    amp - метод усиления сигнала (mean/max) и коэффициент
    without0/1 - использование vad для участков речи/тишины
    old - разметка по целым секундам без смещений на милисекунды
    rttm - использование vad'a Леши и порог склейки
    """

    name = 'AMI'
    if vad == 'test':
        name += '_test'
    name += '_array1-01'
    if audio_amp != 'none':
        name += '_amp' + audio_amp
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

    if add_name_test != '':
        name += '_' + add_name_test

    return name




def predict(sec, features):

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

    # Список параметров
    with open('./yaml/predict.yaml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Изменяемые параметры
    # для настройки: vad, sec, n_classes, features
    v_vad = '2.0'
    args['sec'] = sec
    args['features'] = features

    # только для sizeCN
    #args['n_sec'] = sec
    #args['path_to_file_vad'] = args['path_to_file_vad'] + v_vad + '/'

    model, model_name = load_model(**args)
    #model = model.cuda()
    model.eval()

    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("./models", 'scaler_' + args['features'] + '_AMI.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    audio_names, audio_paths = get_data_list(get_data_names(set_ami='train', **args), **args) # set_ami 'all', 'train', 'test', 'val'

    path_to = args['outputs_path'] + model_name + '/' + create_predict_name(**args) + '/'
    #path_to = 'outputs\\CountNet_CRNN\\' + create_predict_name(**args) + '\\'

    if not os.path.exists(path_to + 'pkl/probs'):
        os.makedirs(path_to + 'pkl/probs')

    # только для sizeCN
    #args['sec'] = 5
    for i in tqdm(range(len(audio_names))): # audio_names.items()
        
        count_audio(audio_names[i], audio_paths[i], model, scaler, path_to, **args) # подсчитывает дикторов на записи и сохраняет результат

    
    #count_audio('TS3003c', audio_names['TS3003c'], model, scaler, path_to, **args)

    count_metrics(audio_names, path_to, **args)
    #count_metrics(audio_names.keys(), path_to, only_predicted=False, **args)
    


def main():
    #sec = [5, 4, 3, 2, 1.5, 1, 0.5]
    #sec = [2, 1.5, 1, 0.5]
    #sec = [3, 4, 5]
    '''
    sec = [1]
    for s in sec:
        predict(s, 'stft')
    #    predict(s, 'melsp')
    #    predict(s, 'mfcc')
    '''
    sec = [2, 1.5, 1]
    for s in sec:
        predict(s, 'stft')
        predict(s, 'mfcc')

    #predict(0.5, 'stft')
    #predict(1, 'stft')
    #predict(1, 'mfcc')



if __name__ == '__main__':
    main()
