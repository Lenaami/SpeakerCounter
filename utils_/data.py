import os
import datetime
import random as rng
import time
import numpy as np
import pickle

classes = {
    4: '1-4',
    5: '0-4'
}

def load_train_data(n_classes, sec, features, use_for_validation, **kwargs):

    t_start = time.time()
    
    with open('data/data_' + create_name(n_classes, sec, features, **kwargs) + '.pickle', 'rb') as f:
        inputs = pickle.load(f)

    with open('data/labels_' + create_name(n_classes, sec, features, forLabels=True, **kwargs) + '.pickle', 'rb') as f:
        outputs = pickle.load(f)
    
    # Experiment Parameters
    n_classes = outputs.shape[1]
    sz_set = inputs.shape[0]
    sz_validate = int(sz_set * use_for_validation)
    sz_train = int(sz_set - sz_validate)
    sz_input = str(inputs.shape[1]) + ', ' + str(inputs.shape[2])

    t_stop = time.time()

    # Debug Messages
    print("Input prepare time   : ", str(t_stop - t_start))
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))
    print("Used for training    : ", str(sz_train))
    print("Used for validation  : ", str(sz_validate))

    ###########################################################################
    # Split Data into Train and Validate
    ###########################################################################

    indexes = np.arange(sz_set)
    np.random.shuffle(indexes)
    
    rand_inputs = []
    rand_outputs = []
    for idx in indexes:
        rand_inputs.append(inputs[idx][:][:]) 
        rand_outputs.append(outputs[idx][:])

    rand_inputs = np.asarray(rand_inputs)
    rand_outputs = np.asarray(rand_outputs)

    x_train = rand_inputs[0:sz_train][:][:]
    x_validate = rand_inputs[sz_train:(sz_train+sz_validate)][:][:]
    y_train = rand_outputs[0:sz_train][:]
    y_validate = rand_outputs[sz_train:(sz_train+sz_validate)][:]

    # Expanding dimensions to be able to use Conv 1D

    x_train = np.expand_dims(x_train, axis = 1)
    x_validate = np.expand_dims(x_validate, axis = 1)

    return x_train, y_train, x_validate, y_validate


def get_list_of_data(path_audio, vad, path_to_file_vad, path_to_file_ideal_vad, **kwargs):

    if vad == 'vad':
        files = os.listdir(path_to_file_vad)
    else: # для 'ideal_vad', 'none'
        files = os.listdir(path_to_file_ideal_vad)
    
    names = [f.split('.')[0] for f in files]

    audio_names = {}

    for d, dirs, files in os.walk(path_audio): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            name = f.split('.')[0]
            if name in names:
                audio_names[name] = path # добавление адреса в список

    return audio_names


def create_name(n_classes, sec, features, forLabels = False, **kwargs):

    if forLabels:
        name = classes[n_classes] + '_' + str(sec) + 'sec'
    else:
        name = classes[n_classes] + '_' + str(sec) + 'sec_' + features

    return name