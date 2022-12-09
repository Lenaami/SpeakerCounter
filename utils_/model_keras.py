# coding=utf-8

import os
import tensorflow as tf
from tensorflow import keras

classes = {
    4: '1-4',
    5: '0-4'
}


def create_model_LSTM(n_classes, sec, features, **kwargs):
    '''
    n_classes = kwargs['n_classes']  
    sec = kwargs['sec']  
    features = kwargs['features']
    '''

    units_reshape_1 = {
        500: 53,
        400: 42,
        300: 31,
        200: 20,
        150: 15,
        100: 9,
        50: 3
    }

    units_reshape_2 = {
        'stft': 1280,
        'melsp': 768,
        'mfcc': 192
    }

    input_features = {
        'stft': 201,
        'melsp': 128,
        'mfcc': 40
    }

    input_length = int(100 * sec)
    n_input_features = input_features[features]  # 201 - stft, 128 - melsp, 40 - mfcc
    n_units_reshape_1 = units_reshape_1[input_length]
    n_units_reshape_2 = units_reshape_2[features]
    n_units_dense = n_classes

    the_network = keras.Sequential()

    the_network.add(keras.layers.Conv2D(filters=64,
                                        kernel_size=3,
                                        strides=(1, 1),
                                        #data_format='channels_first',
                                        input_shape=(input_length, n_input_features, 1))) # изменен формат данных на NHWC

    the_network.add(keras.layers.Conv2D(filters=32,
                                        kernel_size=3,
                                        strides=(1, 1)))#,
                                        #data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(3, 3)))#, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=128,
                                        kernel_size=3,
                                        strides=(1, 1)))#,
                                        #data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=64,
                                        kernel_size=3,
                                        strides=(1, 1)))#,
                                        #data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=(3, 2), strides=(3, 3)))#, data_format='channels_first'))

    the_network.add(keras.layers.Dropout(0.3))

    #the_network.add(keras.layers.Permute((2, 1, 3)))

    the_network.add(
        keras.layers.Reshape((n_units_reshape_1, n_units_reshape_2)))  # 1280 - stft, 768 - melsp, 192 - mfcc

    the_network.add(keras.layers.LSTM(40, return_sequences=True, activation='tanh',
                                      recurrent_activation='sigmoid'))  # activation='tanh', recurrent_activation = 'sigmoid'
    # batch_input_shape=(1, n_units_reshape, 1280), 

    the_network.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

    the_network.add(keras.layers.Flatten())

    the_network.add(keras.layers.Dense(n_units_dense, activation='softmax'))

    return the_network


def create_model_CNN(n_classes, sec, features, **kwargs):
    '''
    n_classes = kwargs['n_classes']  
    sec = kwargs['sec']  
    features = kwargs['features']
    '''

    units_reshape_1 = {
        500: 53,
        400: 42,
        300: 31,
        200: 20,
        150: 15,
        100: 9,
        50: 3
    }

    input_features = {
        'stft': 201,
        'melsp': 128,
        'mfcc': 40
    }

    input_length = int(100 * sec)
    n_input_features = input_features[features]  # 201 - stft, 128 - melsp, 40 - mfcc
    # n_units_reshape_1     = units_reshape_1[input_length]
    # n_units_reshape_2     = units_reshape_2[features]
    n_units_dense = n_classes

    n_filters_L1 = 16  # 32
    n_filters_L2 = 32  # 64
    n_filters_L3 = 64  # 128
    n_pool_sz_L1 = 1
    n_pool_sz_L2 = 2
    n_pool_sz_L3 = 1
    n_kernel_sz_L1 = 6
    n_kernel_sz_L2 = 4
    n_kernel_sz_L3 = 2
    n_strides_L1 = 1
    n_strides_L2 = 1
    n_strides_L3 = 1
    n_units_dense_L1 = 512  # 1024
    n_units_dense_L2 = 256  # 512
    n_units_dense_L3 = 128  # 256
    f_dropout_prob_Lconv = 0.75
    f_dropout_prob_L1 = 0.1
    f_dropout_prob_L2 = 0.1
    f_dropout_prob_L3 = 0.5

    the_network = keras.Sequential()

    # First Block

    the_network.add(keras.layers.Conv2D(filters=n_filters_L1,
                                        kernel_size=n_kernel_sz_L1,
                                        strides=n_strides_L1,
                                        data_format='channels_first',
                                        input_shape=(1, input_length, n_input_features)))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L1, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L1,
                                        kernel_size=n_kernel_sz_L1,
                                        strides=n_strides_L1,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L1, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L1,
                                        kernel_size=n_kernel_sz_L1,
                                        strides=n_strides_L1,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L1, data_format='channels_first'))

    # Second Block

    the_network.add(keras.layers.Conv2D(filters=n_filters_L2,
                                        kernel_size=n_kernel_sz_L2,
                                        strides=n_strides_L2,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L2, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L2,
                                        kernel_size=n_kernel_sz_L2,
                                        strides=n_strides_L2,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L2, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L2,
                                        kernel_size=n_kernel_sz_L2,
                                        strides=n_strides_L2,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L2, data_format='channels_first'))

    # Third Block

    the_network.add(keras.layers.Conv2D(filters=n_filters_L3,
                                        kernel_size=n_kernel_sz_L3,
                                        strides=n_strides_L3,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L3, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L3,
                                        kernel_size=n_kernel_sz_L3,
                                        strides=n_strides_L3,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L3, data_format='channels_first'))

    the_network.add(keras.layers.Conv2D(filters=n_filters_L3,
                                        kernel_size=n_kernel_sz_L3,
                                        strides=n_strides_L3,
                                        data_format='channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=n_pool_sz_L3, data_format='channels_first'))

    #

    the_network.add(keras.layers.Dropout(f_dropout_prob_Lconv))

    the_network.add(keras.layers.BatchNormalization())

    the_network.add(keras.layers.Flatten())

    the_network.add(keras.layers.Dense(n_units_dense_L1, activation='sigmoid'))

    the_network.add(keras.layers.Dropout(f_dropout_prob_L1))

    the_network.add(keras.layers.Dense(n_units_dense_L2, activation='relu'))

    the_network.add(keras.layers.Dropout(f_dropout_prob_L2))

    the_network.add(keras.layers.Dense(n_units_dense_L3, activation='relu'))

    the_network.add(keras.layers.Dropout(f_dropout_prob_L3))

    the_network.add(keras.layers.Dense(n_units_dense, activation='softmax'))

    return the_network


def load_model(**kwargs):  # model_arch, n_classes, sec, features, dataset, add_name

    model_name = create_model_name(**kwargs)

    if not os.path.exists(kwargs['checkpoints_path'] + model_name + '/the_network.h5'):

        if not os.path.exists(kwargs['checkpoints_path'] + model_name):
            os.makedirs(kwargs['checkpoints_path'] + model_name)

        if kwargs['model_arch'] == 'LSTM':
            the_network = create_model_LSTM(**kwargs)
        elif kwargs['model_arch'] == 'CNN':
            the_network = create_model_CNN(**kwargs)

        # Можно добавить метрики в создание модели
        the_network.compile(optimizer=tf.optimizers.Adam(kwargs['start_lr']),
                            loss=keras.losses.categorical_crossentropy,
                            metrics=['categorical_accuracy'])

    else:
        the_network = keras.models.load_model(
            os.path.join(kwargs['checkpoints_path'] + model_name, 'the_network.h5'),
        )

    return the_network, model_name


def create_model_name(model_arch, n_classes, sec, features, dataset, add_name, **kwargs):
    '''
    model_arch = kwargs['model_arch']
    n_classes = kwargs['n_classes']  
    sec = kwargs['sec']  
    features = kwargs['features'] 
    dataset = kwargs['dataset'] 
    add_name = kwargs['add_name']
    '''

    name = 'model_' + model_arch + '_' + classes[n_classes] + '_' + str(sec) + 'sec_' + features + '_' + dataset
    if add_name != '':
        name = name + '_' + add_name

    return name
