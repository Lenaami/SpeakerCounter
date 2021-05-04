
import numpy as np
import soundfile as sf
import librosa
import pickle
import pandas as pd

path_audio = 'F:/amicorpus_1'
v_vad = '2.0'

#file_csv = 'outputs\\AMIcsv\\'
#file_rttm = 'outputs\\logunov\\speech_regions_predicted_vad_markup_bad_repaired_1.0\\'
#SEC = 1
#path_to = 'AMI_amp4mean_without0_old_rttm10_array1-01' 
#model (номер)
#classes (4/5)

length = {}

def AMP_audio(audio, rate):

    seconds = len(audio)//rate 

    amp = 0
    X_max = []
    for i in range(seconds):
        X = np.abs(librosa.stft(audio[i*rate:(i+1)*rate], n_fft=400, hop_length=160)).T
        X_max.append(np.max(X))
        #if np.max(X) < 5 and AMP < np.max(X):
        #    AMP = np.max(X)

    if np.mean(X_max) < 4:
        amp = 4/(np.mean(X_max) + 0.001)
    else:
        amp = 1

    return amp

def count(audio, model, scaler, sec, AMP):

    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    X_0 = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    ####### ТУПОЕ УСИЛЕНИЕ АМПЛИТУДЫ
    '''
    if np.max(X_0) > 5:
        AMP = 1    
    else:
        AMP = AMP
        #AMP = 5/(np.max(X_0) + 0.001) # Поиграть с этой величиной / попробовать умножать на статичный коэф. (4?)
    '''
    X_0 *= AMP
    
    #######

    # apply global (featurewise) standardization to mean1, var0
    X_0 = scaler.transform(X_0)

    # cut to input shape length (500 frames x 201 STFT bins)
    X_0 = X_0[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!
    X = np.zeros((100, 201))

    X[:X_0.shape[0], :] = X_0 

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]


def count_part(audio, rate, seconds, model, scaler, sec, amp, features, **kwargs):

    eps = np.finfo(np.float).eps
    # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    X_part = []

    #print(seconds)

    for i in range(len(seconds)):
        sec_begin = int(seconds[i]*rate)
        audio_part = audio[sec_begin:sec_begin+int(sec*rate)]
        
        
        # другие признаки
        #print(audio_part.shape, '\t', sec_begin, sec_begin+int(sec*rate))

        X_0 = np.abs(librosa.stft(audio_part, n_fft=400, hop_length=160)).T
        #X_0 *= amp

        if features == 'stft':

            #X_0 *= amp
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 201))

        if features == 'melsp_1':

            X_0 = librosa.feature.melspectrogram(y=audio_part, sr=rate, S=X_0.T, n_fft=400, hop_length=160).T
            X_0 = librosa.power_to_db(X_0, ref=np.max)
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 128))

        if features == 'mfcc':

            X_0 = librosa.feature.mfcc(y=audio_part, sr=rate, S=X_0.T, n_mfcc=40).T
            X_0 = scaler.transform(X_0)
            X = np.zeros((int(100*sec), 40))

    #######

    # apply global (featurewise) standardization to mean1, var0
        # обучить свой ???
        #X_0 = scaler.transform(X_0)

    # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = X_0[:int(100*sec), :] 
        X[:X_0.shape[0], :] = X_0 

    # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        X_part.append(X)

    # add sample dimension
    #X = X[np.newaxis, ...]
    
    X_part = np.asarray(X_part)

    if len(model.input_shape) == 4:
        X_part = X_part[:, np.newaxis, ...]

    ys = model.predict(X_part, verbose=0)
    return ys
    #return np.argmax(ys, axis=1)

def count_probs():

    # вывод вероятностей
        # Усилить амплитуду fft и преобразовать обратно (????)

    # compute STFT
    X_part = []

    for i in range(len(seconds)):
        sec_begin = int(seconds[i]*rate)
        audio_part = audio[sec_begin:sec_begin+sec*rate]
        
        # другие признаки
        X_0 = np.abs(librosa.stft(audio_part, n_fft=400, hop_length=160)).T

        X_0 *= AMP
    
    #######

    # apply global (featurewise) standardization to mean1, var0
        # обучить свой ???
        X_0 = scaler.transform(X_0)

    # cut to input shape length (500 frames x 201 STFT bins)
        X_0 = X_0[:int(100*sec), :] # ИЗМЕНЕНО С 500 !!!!!!!!!
        X = np.zeros((100, 201))

        X[:X_0.shape[0], :] = X_0 

    # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        X_part.append(X)

    # add sample dimension
    #X = X[np.newaxis, ...]
    
    X_part = np.asarray(X_part)

    if len(model.input_shape) == 4:
        X_part = X_part[:, np.newaxis, ...]

    ys = model.predict(X_part, verbose=0)
    return np.argmax(ys, axis=1)


def get_seconds_VAD(name, vad, sec, inverse_vad, ms, path_to_file_vad, path_to_file_ideal_vad, path_to_file_test, **kwargs): # inverse - взять те промежутки, где VAD НЕ отметил речь

    if vad == 'vad':
        # Использование VADа
        file = path_to_file_vad + name + '.segments.rttm'

        data = []
        with open(file, encoding='utf-8') as r_file:
            for line in r_file:
                ln = line.split(' ')
                data.append([float(ln[2]), float(ln[3])])

    else:
        # Идеальная разметка
        if vad == 'ideal_vad':
            file = path_to_file_ideal_vad + name + '.no_silence.csv' # путь к файлам

        # Тестовая разметка
        if vad == 'test':
            file = path_to_file_test + name + '.test_count.csv'

        DATA = pd.read_csv(file, delimiter=',') # колонки - start, duration
        data = DATA.values

   
    seconds = []

    # с начала (которое указано), по одной секунде, оставшийся отрезок меньше 0.1-0.2?, -> нужные секунды
    # пример 11.09, 4.4399999999999995 -> 11.09, 12.09, 13.09, 14.09, 15.09 --> далее перечисление продолжается со следующего промежутка

    if ms:
        # Разметка по миллисекундам
        for st, dur in data:
            times = dur // sec + (dur % sec > float(sec) / 5)# длительность последнего фрагмента должна превышать 0,2 от секунд обработки
        
            for i in range(int(times)):
                seconds.append(st + i*sec)
   
    else:
        # Сравнение с общей разметкой по 1 секунде
        for st, dur in data:
            times = (dur + (st - int(st))) // sec + ((dur + (st - int(st))) % sec > 0) # вычисление кол-ва секунд, которые нужно обработать
                
            for i in range(times): # int(times) ?????
                seconds.append(int(st) + i*sec)    


    seconds = list(set(seconds)) # убираем повторения (если есть)

    if inverse_vad == True:        
        seconds_all = list(np.arange(length[name])) # список секунд (нужно знать длину файлов)  ### НЕ ARANGE А, I*SEC
        return [s for s in seconds_all if s not in seconds] # not in - участки без речи, in участки с речью

    return seconds  


def count_audio(name, path, model, scaler, path_to, vad, sec, **kwargs):

    # взять файл и предсказать значения для него
    # использование vad, rttm


    #t_file = time.time()

    # compute audio
    audio, rate = sf.read(path, always_2d=True)

    #print(audio.shape)

    #length[name] = (len(audio)//rate)//sec
        
    # downmix to mono
    audio = np.mean(audio, axis=1)

    amp = AMP_audio(audio, rate)
        
    if vad != 'none':
        seconds = get_seconds_VAD(name, vad, sec, **kwargs) # использование vad
    else:
        length = int((len(audio)//rate)//sec) + ((len(audio)//rate) % sec > 0) # не учитывается остаток отрезка (если есть)  - делим аудио на отрезки (это их количество)
        seconds = [i*sec for i in range(0, length)] # все секунды подаем на вход

    ##### Партия
    part = int(2000 // sec) # длина партии
    sec_parts = len(seconds) // part + (len(seconds) % part > 0) # количество партий

    estimate = []
    for i in range(sec_parts):
        estimate.extend(count_part(audio, rate, seconds[i*part:(i+1)*part], model, scaler, sec, amp, **kwargs))

    with open(path_to + 'pkl\\' + name + '.output_probs.pickle', 'wb') as f:
        pickle.dump(estimate, f)

    estimate = np.asarray(estimate)
    if estimate.shape[0] == 0:
        estimate == []
    else:
        estimate = np.argmax(estimate, axis=1)


    with open(path_to + 'pkl\\' + name + '.output.pickle', 'wb') as f:
        pickle.dump(estimate, f)
    


    #print(name_f[k])
    #t_stop = time.time()
    #print("Time: " + str(t_stop - t_start))
    #print("File: " + str(t_stop - t_file))


    # можно принимать до 2000 секунд