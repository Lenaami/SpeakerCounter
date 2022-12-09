# coding=utf-8

import numpy as np
import pickle

from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sn

from utils_.count import get_seconds_VAD

classes = {
    4: '1-4',
    5: '0-4'
}

def MAE(true_labels, pred_labels):

    mae_score = []

    n_classes = len(set(true_labels))

    predicted = [[] for i in range(n_classes)]

    for i in range(len(true_labels)):
        if n_classes == 4:
            predicted[int(true_labels[i])-1].append(int(pred_labels[i])-1)
        else:
            predicted[int(true_labels[i])].append(int(pred_labels[i]))

    for i in range(n_classes):
        mae_score.append(np.mean(np.abs(i - np.asarray(predicted[i]))))

    return np.mean(mae_score), np.max(mae_score), np.min(mae_score), mae_score


def get_labels_AMI(name, vad, sec, inverse_vad, path_to_file_count, **kwargs):

    file_csv = path_to_file_count + name + '.count.csv' # путь к файлам

    DATA = pd.read_csv(file_csv, delimiter=',') # столбцы - start, duration, count или (start, end, duration, count)
    data = DATA.values
    data[:, 1] = data[:, 0] + data[:, 1] # переделываю в start, end, count

    count = []

    if vad != 'none':
        seconds = get_seconds_VAD(name, vad, sec, inverse_vad, **kwargs) # секунды, для которых нужно предсказать кол-во дикторов

        for i in range(len(seconds)):
            persons = [0]
            for st, ed, pers in data:
                if (seconds[i] + sec - st) > 0 and (ed - seconds[i]) > 0:
                    persons.append(pers)
            count.append(max(persons))
    else:
        # разметка по секундам целого файла
        for i in range(int(data[-1, 1] // sec + 1)):
            persons = [0]
            for st, ed, pers in data:
                if ((i+1)*sec - st) > 0 and (ed - i*sec) > 0:
                    persons.append(pers)
            count.append(max(persons))

    return count


def read_result(name, path_to, only_predicted, n_classes, sec, vad, v_vad, inverse_vad, **kwargs):

    #if only_predicted == False:
    #    true_labels = get_labels_AMI(name, vad='none', sec, **kwargs)
    #else:
    true_labels = get_labels_AMI(name=name, vad='none', sec=sec, inverse_vad=inverse_vad, **kwargs) # только для vad по целым секундам
    #true_labels = get_labels_AMI(name, vad, sec, inverse_vad, **kwargs)

    with open(path_to + 'pkl/' + name + '.output.pickle', 'rb') as f:
        pr_labels = pickle.load(f)
        
  
    #seconds = get_seconds_VAD(name, vad, sec, **kwargs)

    '''
    # только для vad по целым секундам
    if vad != 'none':
        seconds = get_seconds_VAD(name, vad, sec, inverse_vad, **kwargs) # использование vad
    else:
        length = len(pr_labels) #(len(audio)//rate)//SEC 
        seconds = [i*sec for i in range(0, length)] # все секунды подаем на вход


    pred_labels = []
    #for i in range(0, seconds[-1] + 1, sec):
    for i in range(0, int(seconds[-1] // sec) + 1):
        #if i in seconds:
        if i*sec in seconds:
            #print(i*sec)
            #pred_labels.append(pr_labels[seconds.index(i)] + 1)
            if sec == 0.5:
                pred_labels.append(pr_labels[seconds.index(int(i*sec))])
            else:
                pred_labels.append(pr_labels[seconds.index(i)])
        else:
            pred_labels.append(0)
            #pred_labels.append(1) # вместо речи
    '''

    pred_labels = pr_labels

    true = [0]*max(len(pred_labels), len(true_labels))
    pred = [0]*max(len(pred_labels), len(true_labels))
    true[:len(true_labels)] = true_labels[:]
    pred[:len(pred_labels)] = pred_labels[:]

    ### Классификация для разного количества классов
    
    num_cl = 4 # до n+
    
    for i in range(len(true)):
        if true[i] > num_cl:
            true[i] = num_cl
        if pred[i] > num_cl:
            pred[i] = num_cl
    
    ###

    return true, pred

def count_metrics(audio_names, path_to, n_classes, only_predicted = True, **kwargs):

    f1 = [] 

    true_l = []
    pred_l = []

    for name in audio_names:
        true, pred = read_result(name, path_to, only_predicted, n_classes, **kwargs)

        true_l.extend(true)
        pred_l.extend(pred)

        f1.append(f1_score(true, pred, average='weighted'))

    if only_predicted:
        add_name = '_predicted'
        #add_name = '_class0-3'
    else:
        add_name = ''

    print('Среднее по датасету')
    print(np.mean(f1))

    print('Объединенные результаты по датасету')
    print(f1_score(true_l, pred_l, average='weighted'))

    print('F1-score (micro): ', f1_score(true_l, pred_l, average='micro'))
    print('F1-score (macro): ', f1_score(true_l, pred_l, average='macro'))

    print('Accuracy: ', accuracy_score(true_l, pred_l))

    mae, max_mae, min_mae, mae_score = MAE(true_l, pred_l)

    print('MAE: ', mae, ' +- ', max_mae - mae)
    print('MAE max: ', max_mae)
    print('MAE min: ', min_mae)
    print('MAE score: ', mae_score)

    print(metrics.classification_report(true_l, pred_l))

    print('Confusion matrix (Precision)')
    print(metrics.confusion_matrix(true_l, pred_l, normalize='pred'))  # 'true' - recall, 'pred' - precisoin, 'all'
    print('Confusion matrix (Recall)')
    print(metrics.confusion_matrix(true_l, pred_l, normalize='true'))

    '''
    if n_classes == 4: # прописать больше (или просто посчитать кол-во классов, наличие 0 и тп)
        indexs = [i for i in range(1, 11)]
    else:
        indexs = [i for i in range(11)]
    '''
    indexs = [i for i in range(11)] # при использовании vad

    cm = metrics.confusion_matrix(true_l, pred_l, normalize='pred') # 'true', 'pred', 'all'
    rg = cm.shape[0]
    if rg == 4:
        indexs = [i for i in range(1, 11)]

    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (6,4))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig(path_to + 'conf_matrix_precision' + add_name + '.png')

    cm = metrics.confusion_matrix(true_l, pred_l, normalize='true') # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (6,4))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig(path_to + 'conf_matrix_recall' + add_name + '.png')

    cm = metrics.confusion_matrix(true_l, pred_l) # 'true', 'pred', 'all'
    df_cm = pd.DataFrame(cm, index = indexs[:rg], columns = indexs[:rg])
    fig = plt.figure(figsize = (6,4))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap="Purples") # Oranges, YlGnBu, Blues, Purples,  fmt='d'
    #plt.show()
    fig.savefig(path_to + 'conf_matrix' + add_name + '.png')

    with open(path_to + 'report' + add_name + '.txt', 'w') as f:
        f.write('F1-score (weighted)\n')
        f.write('Среднее по датасету: ' + str(np.mean(f1)) + '\n')
        f.write('Объединенные результаты по датасету: ' + str(f1_score(true_l, pred_l, average='weighted')) + '\n')
        f.write('F1-score (micro): ' + str(f1_score(true_l, pred_l, average='micro')) + '\n')
        f.write('F1-score (macro): ' + str(f1_score(true_l, pred_l, average='macro')) + '\n')
        f.write('Accuracy: ' + str(accuracy_score(true_l, pred_l)) + '\n')
        f.write('MAE: ' + str(mae) + ' +- ' + str(max_mae - mae) + '\n')
        f.write('MAE max: ' + str(max_mae) + '\n')
        f.write('MAE min: ' + str(min_mae) + '\n')
        f.write('MAE score: ' + str(mae_score) + '\n')
        f.write('\n')
        f.write(str(metrics.classification_report(true_l, pred_l)) + '\n')
        f.write('\n')
        f.write('Confusion matrix (Precision)\n')
        f.write(str(metrics.confusion_matrix(true_l, pred_l, normalize='pred')) + '\n')
        f.write('Confusion matrix (Recall)\n')
        f.write(str(metrics.confusion_matrix(true_l, pred_l, normalize='true')) + '\n')


def count_EER():
    return 0