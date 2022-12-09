# coding=utf-8

import os
import time
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.client import device_lib

from utils_.model_keras import load_model
from data_generator_keras import DataGenerator


def train(sec, features):

    # Список параметров
    with open('./yaml/train.yaml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Изменяемые параметры
    bs = int(125 // sec)
    args['batch_size'] = bs
    args['sec'] = sec
    args['features'] = features

    # Сеты AMI
    with open('yaml/ami_sets_mini5.yaml') as f:
        ami_sets = yaml.load(f, Loader=yaml.FullLoader)
    training_generator = DataGenerator(ami_sets['train'], **args)
    validation_generator = DataGenerator(ami_sets['test'], **args)

    # Загрузка/создание модели
    the_network, model_name = load_model(**args)
    the_network.summary()
    
    s_model_save_dir =  args['checkpoints_path'] + model_name

    s_log_file = s_model_save_dir + "/the_network_log.csv"
    csv_logger = CSVLogger(s_log_file, append=True, separator=';')
    
    model_saver = keras.callbacks.ModelCheckpoint(s_model_save_dir + "/the_network.h5",
                                                  monitor='val_categorical_accuracy', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, 
                                                  mode='auto', 
                                                  save_freq='epoch')

    t_start = time.time()

    the_network.fit(
                    training_generator, 
                    epochs=args['n_epochs'], 
                    batch_size=args['batch_size'], 
                    validation_data=validation_generator,
                    callbacks=[csv_logger, model_saver])
    
    
    t_stop = time.time()
    print("Training time : " + str(t_stop - t_start))


def main(_):

    #sec = [5, 4, 3, 2, 1.5, 1, 0.5]
    #sec = [5, 4, 3, 2]
    #sec = [2, 1.5, 1, 0.5]
    sec = [1]
    #features = ['stft', 'melsp_1', 'mfcc']
    features = ['stft']
    for s in sec:
        for feat in features:
            print('cool')
            #train(s, feat)


if __name__ == '__main__':

    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    '''


    print(device_lib.list_local_devices())


    tf.compat.v1.app.run()
