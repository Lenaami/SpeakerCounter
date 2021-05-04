import os
import datetime
import random as rng
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix

import pickle
from tensorflow.python.client import device_lib

# Inputs
x_filename = '/home/valentin_m_andrei/datasets/300ms_fft_env_hist/x_train_normalized.txt'
y_filename = '/home/valentin_m_andrei/datasets/300ms_fft_env_hist/y_train.txt'
#s_model_save_dir = '/home/valentin_m_andrei/checkpoints/'
s_model_save_dir = 'checkpoints/'

# x_filename = '/home/valentin_m_andrei/datasets/x_dummy.txt'
# y_filename = '/home/valentin_m_andrei/datasets/y_dummy.txt'
# s_model_save_dir = '/home/valentin_m_andrei/checkpoints/'

# Architecture
n_filters_L1        = 16
n_filters_L2        = 32
n_kernel_sz_L1      = 16
n_kernel_sz_L2      = 8
n_strides_L1        = 1
n_strides_L2        = 1
n_strides_L3        = 1
n_units_dense_L1    = 2048
n_units_dense_L2    = 1024
n_units_dense_L3    = 512
f_dropout_prob_L1   = 0.8
f_dropout_prob_L2   = 0.1
f_dropout_prob_L3   = 0.1

# Training
f_use_for_validation    = 0.02
sz_batch                = 100 # 25 (10 не оч)
n_epochs                = 160
#n_epochs                = 1
f_start_lr              = 0.001

# Plotting & debugging
# TODO

class ConfusionMatrix(keras.callbacks.Callback):

    def __init__(self, val_data):
        super(ConfusionMatrix, self).__init__()
        self.validation_data = val_data

    def on_epoch_end(self, epoch, logs={}):
        print('Confusion matrix: ')
        y_prob = self.model.predict(self.validation_data[0])
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(self.validation_data[1], axis=1)
        print(confusion_matrix(y_true, y_pred))

def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    with open('data/data_1-4_1sec.pickle', 'rb') as f:
        #inputs_1 = pickle.load(f)
        inputs = pickle.load(f)
    
    #with open('data/data_2.pickle', 'rb') as f:
        #inputs_2 = pickle.load(f)

    #inputs = np.vstack((inputs_1, inputs_2))

    with open('data/labels_1-4_1sec.pickle', 'rb') as f:
        outputs = pickle.load(f)
    #inputs = np.loadtxt(x_filename)
    #outputs = np.loadtxt(y_filename)
    outputs = outputs[:inputs.shape[0]][:]

    # Experiment Parameters
    n_classes = outputs.shape[1]
    sz_set = inputs.shape[0]
    sz_validate = int(sz_set * f_use_for_validation)
    #sz_train = int(sz_set * 0.1)
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

    #x_train = inputs[0:sz_train][:]
    #x_validate = inputs[sz_train:sz_set][:]
    #y_train = outputs[0:sz_train][:]
    #y_validate = outputs[sz_train:sz_set][:]

    indexes = np.arange(sz_set)
    np.random.shuffle(indexes)
    
    rand_inputs = []
    rand_outputs = []
    for idx in indexes:
        rand_inputs.append(inputs[idx][:][:]) ### [idx][:500][:]
        rand_outputs.append(outputs[idx][:])

    rand_inputs = np.asarray(rand_inputs)
    rand_outputs = np.asarray(rand_outputs)

    #print(rand_inputs.shape)
    #print(rand_outputs.shape)

    x_train = rand_inputs[0:sz_train][:][:] ### [0:sz_train][:][:]
    #x_validate = rand_inputs[sz_train:sz_set][:][:]
    x_validate = rand_inputs[sz_train:(sz_train+sz_validate)][:][:] ### [sz_train:(sz_train+sz_validate)][:][:]
    y_train = rand_outputs[0:sz_train][:]
    #y_validate = rand_outputs[sz_train:sz_set][:]
    y_validate = rand_outputs[sz_train:(sz_train+sz_validate)][:]

    # Expanding dimensions to be able to use Conv 1D

    x_train = np.expand_dims(x_train, axis = 1)
    x_validate = np.expand_dims(x_validate, axis = 1)

    #print(x_train.shape, x_validate.shape)
    #print(y_train.shape, y_validate.shape)


    ############# Создание сетов для обучения (динамические?)

    #############

    #tf.test.is_built_with_cuda()
    #tf.config.list_physical_devices('GPU')



    #print(device_lib.list_local_devices())

    #keras.tensorflow_backend._get_available_gpus()

    #config = tf.ConfigProto()
    #tf.config.gpu.set_per_process_memory_growth = True
    #sess = tf.Session(config=config)

    ###########################################################################
    # Targeted Neural Network Architecture
    ###########################################################################

    the_network = keras.Sequential()

    the_network.add(keras.layers.Conv2D(filters = 64, 
                                        kernel_size = 3, 
                                        strides = (1, 1),
                                        data_format = 'channels_first', 
                                        input_shape=(1, 100, 201))) # input_shape=(1, 500, 201)))

    the_network.add(keras.layers.Conv2D(filters = 32, 
                                        kernel_size = 3, 
                                        strides = (1, 1),
                                        data_format = 'channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(3, 3), data_format = 'channels_first'))

    the_network.add(keras.layers.Conv2D(filters = 128, 
                                        kernel_size = 3, 
                                        strides = (1, 1),
                                        data_format = 'channels_first'))

    the_network.add(keras.layers.Conv2D(filters = 64, 
                                        kernel_size = 3, 
                                        strides = (1, 1),
                                        data_format = 'channels_first'))

    the_network.add(keras.layers.MaxPooling2D(pool_size=(3, 2), strides=(3, 3), data_format = 'channels_first'))                                      

    the_network.add(keras.layers.Dropout(0.3))

    the_network.add(keras.layers.Permute((2, 1, 3)))

    the_network.add(keras.layers.Reshape((9, 1280))) # (53, 1280) для длины 500 ; (42, 1280) для длины 400 ; (31, 1280) для длины 300 ; (20, 1280) для длины 200 ; (15, 1280) для длины 150 ; (9, 1280) для длины 100 ; 3 для 50

    the_network.add(keras.layers.LSTM(40, return_sequences=True, activation='tanh', recurrent_activation = 'sigmoid')) # activation='tanh', recurrent_activation = 'sigmoid'

    the_network.add(keras.layers.MaxPooling1D(pool_size=2, strides=2)) 

    the_network.add(keras.layers.Flatten())

    #the_network.add(keras.layers.Dense(11, activation='softmax'))
    the_network.add(keras.layers.Dense(4, activation='softmax'))

    the_network.summary()

    ###########################################################################
    # Run Training
    ###########################################################################

    #'''
    the_network.compile(optimizer=tf.optimizers.Adam(f_start_lr), 
                        loss=keras.losses.categorical_crossentropy, 
                        metrics=['categorical_accuracy'])

    the_network.summary()

    s_log_file = s_model_save_dir + "the_network_log.csv"

    class_predictions = ConfusionMatrix((x_validate, y_validate))

    csv_logger = CSVLogger(s_log_file, append=True, separator=';')
    
    model_saver = keras.callbacks.ModelCheckpoint(s_model_save_dir + "the_network.h5", 
                                                  monitor='val_categorical_accuracy', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, 
                                                  mode='auto', 
                                                  save_freq='epoch') #period=1

    t_stiart = time.time()
    the_network.fit(x = x_train, 
                    y = y_train, 
                    epochs=n_epochs, 
                    batch_size=sz_batch, 
                    validation_data=(x_validate, y_validate),
                    callbacks=[class_predictions, csv_logger, model_saver])
                    #callbacks=[csv_logger, model_saver])
    
    t_stop = time.time()
    print("Training time : " + str(t_stop - t_start))
    
    #'''

if __name__ == '__main__':

    tf.compat.v1.app.run()