import os
import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix

from utils_.model import create_model_LSTM, create_model_CNN, load_model, create_model_name
from utils_.data import load_train_data

from data_generator import DataGenerator

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

def train(sec, features):

    # задать список параметров, которые можно менять (длительность, модель, классы, предобработка)
    sb = int(125 // sec) #int(125 // sec)

    args = {
        'f_use_for_validation': 0.02,
        'batch_size': sb, # 25 для 5 секунд (10 не оч)
        'n_epochs': 160, #160
        'f_start_lr': 0.001,


        'model_arch': 'LSTM', # 'CNN', 'LSTM'
        'n_classes': 5,
        'sec': sec, 
        'features': features, # 'stft', 'melsp', 'melsp_1', 'mfcc'

        'dataset': 'AMI', # 'LC', 'AMI', 'LC_AMI'
        'audio_path': 'F:\\amicorpus_1\\'
    }
 
    s_model_save_dir =  'checkpoints\\' + create_model_name(**args)

    if not os.path.exists(s_model_save_dir):
        os.makedirs(s_model_save_dir)

    s_log_file = s_model_save_dir + "\\the_network_log.csv"


    #x_train, y_train, x_validate, y_validate = load_train_data(**args)

    
    train_audio_list = ['EN2001a']
    validation_audio_list = ['EN2001b']
    training_generator = DataGenerator(train_audio_list, **args)
    validation_generator = DataGenerator(validation_audio_list, **args)

    the_network = create_model_LSTM(**args)

    #the_network = create_model_CNN(**args)

    
    the_network.compile(optimizer=tf.optimizers.Adam(args['f_start_lr']), 
                        loss=keras.losses.categorical_crossentropy, 
                        metrics=['categorical_accuracy'])

    the_network.summary()
    
    #class_predictions = ConfusionMatrix((x_validate, y_validate))

    csv_logger = CSVLogger(s_log_file, append=True, separator=';')
    
    model_saver = keras.callbacks.ModelCheckpoint(s_model_save_dir + "\\the_network.h5", 
                                                  monitor='val_categorical_accuracy', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, 
                                                  mode='auto', 
                                                  save_freq='epoch')

    t_start = time.time()
    '''
    the_network.fit(x = x_train, 
                    y = y_train, 
                    epochs=args['n_epochs'], 
                    batch_size=args['batch_size'], 
                    validation_data=(x_validate, y_validate),
                    callbacks=[class_predictions, csv_logger, model_saver])
    '''
    # Для DataGenerator (не придумано как переделать class predictions) 
    the_network.fit(
                    training_generator, 
                    epochs=args['n_epochs'], 
                    batch_size=args['batch_size'], 
                    validation_data=validation_generator,
                    callbacks=[csv_logger, model_saver])
    
    
    t_stop = time.time()
    print("Training time : " + str(t_stop - t_start))
    


    # сделать такой же код для train + дообучение (попробовать подгрузить модель, скомпилировать и запустить дообучение)
    # сделать код для графиков (*)
    # функция для создания датасета (разные длительности и разная предобработка)



def main(_):

    #sec = [5, 4, 3, 2, 1.5, 1, 0.5]
    #sec = [5, 4, 3, 2]
    #sec = [2, 1.5, 1, 0.5]
    sec = [1]
    #features = ['stft', 'melsp_1', 'mfcc']
    features = ['stft']
    for s in sec:
        for feat in features:
            train(s, feat)


if __name__ == '__main__':

    
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
    '''
    config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    '''

    tf.compat.v1.app.run()
