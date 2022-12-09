# coding=utf-8
#############
# Команда для запуска
# python output.py f:/amicorpus_1 --model 10
#############

import numpy as np
import soundfile as sf
import argparse
import os
import sklearn
import librosa

import pickle

from sklearn.metrics import f1_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn


def create_csv_out(sec, name_test):
  

    #name_test = 'model_CNN_0-4_1sec_stft_LC/AMI_array1-01_amp4mean'
    save_path_pkl = './outputs/' + name_test + '/pkl/'
    save_path_pkl_probs = './outputs/' + name_test + '/pkl/probs/'
    #save_path_csv = './outputs/' + name_test + '/csv/'
    save_path_rttm = './outputs/' + name_test + '/rttm/'
    save_path_rttm_probs = './outputs/' + name_test + '/rttm_probs/'

    #save_path_public = '/srv/data/NIRMA/yevseyeva/speaker_counter/' + name_test.split('/')[0] + '/csv/'
    
    #sec = 1

    name_f = []
    for d, dirs, files in os.walk(save_path_pkl): # путь к файлам
        for f in files: 
            name_f.append(f.split('.')[0])


    #if not os.path.exists(save_path_csv):
    #    os.makedirs(save_path_csv)

    if not os.path.exists(save_path_rttm):
        os.makedirs(save_path_rttm)

    if not os.path.exists(save_path_rttm_probs):
        os.makedirs(save_path_rttm_probs)

    #if not os.path.exists(save_path_public):
    #    os.makedirs(save_path_public)

    for name in name_f:

        with open(save_path_pkl + name + '.output.pickle', 'rb') as f:
            pr_labels = pickle.load(f)

        with open(save_path_pkl_probs + name + '.output_probs.pickle', 'rb') as f:
            pr_probs = pickle.load(f)

        # Для всех данных (учитывать модельку 0-4, 1-4)
        
        #pred_labels = []
        #for i in range(len(pr_labels)):
        #    #pred_labels.append(pr_labels[i] + 1)  
        #    pred_labels.append(pr_labels[i])
         

        output = []
        for i in range(len(pr_labels)):
            output.append([i*sec, sec, pr_labels[i]])

        output_probs = []
        for i in range(len(pr_probs)):
            output_probs.append([i*sec, sec, pr_probs[i]])


        #OUT = pd.DataFrame(output, columns=['start', 'duration', 'count'])
        #OUT = pd.DataFrame(output, columns=['start', 'end', 'speakers'])
        #OUT.to_csv(save_path_csv + name + '.SCout.csv', index=False)
        #OUT.to_csv(save_path_public + name + '.SCout.csv', index=False)

        array = []
        array_probs = []

        #SPEECH EN2001a 2.91 0.03 формат

        for st, dur, sp in output:
            array.append('SPEECH ' + name + ' ' + str(st) + ' ' + str(dur) + ' ' + str(sp) + '\n') # sp - количество спикеров

        for st, dur, sp in output_probs:
            array_probs.append('SPEECH ' + name + ' ' + str(st) + ' ' + str(dur) + ' ' + str(sp[0]) + ' ' + str(sp[1]) + ' ' + str(sp[2]) + ' ' + str(sp[3]) + ' ' + str(sp[4]) + '\n') # sp - вероятности


        with open(save_path_rttm + name + '.SCout.rttm', 'w') as f:
            f.writelines(array)

        with open(save_path_rttm_probs + name + '.SCout_probs.rttm', 'w') as f:
            f.writelines(array_probs)



if __name__ == '__main__':

    tests = [
        [1, 'model_LSTM_0-4_1sec_stft_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [1.5, 'model_LSTM_0-4_1.5sec_stft_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [2, 'model_LSTM_0-4_2sec_stft_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [1, 'model_LSTM_0-4_1sec_mfcc_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [1.5, 'model_LSTM_0-4_1.5sec_mfcc_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [2, 'model_LSTM_0-4_2sec_mfcc_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [0.5, 'model_CNN_0-4_0.5sec_stft_AMIaug_mini20_scalerAMI_aug4/AMI_array1-01'],
        [2, 'model_LSTM_0-4_2sec_mfcc_AMIaug_mini20_scalerAMI_augMH/AMI_array1-01']
    ]

    #name_test = 'model_CNN_0-4_1sec_stft_LC/AMI_array1-01_amp4mean'
    #sec = 1

    for sec, name_test in tests:
        create_csv_out(sec, name_test + '_train')
        create_csv_out(sec, name_test + '_val')
        create_csv_out(sec, name_test + '_alex')