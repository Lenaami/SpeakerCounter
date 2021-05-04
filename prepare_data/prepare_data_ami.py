import numpy as np
import soundfile as sf
import os
import sklearn
import librosa
import pickle

eps = np.finfo(np.float).eps

def prepare_data(path_f, names_f, sec, feature):

    
    #LABELS = np.zeros((len(path_f), 11))
    #LABELS = np.zeros((520*int(5 // sec)*5, 5)) # 1 - кол-во примеров в классе, 2 - кол-во частей в 5 секундах(?), 3 - кол-во классов

    for k in range(len(path_f)):

        DATA = []
    # compute audio
        audio, rate = sf.read(path_f[k], always_2d=True)

    # downmix to mono
        audio = np.mean(audio, axis=1)
        #seconds = (len(audio)//rate)//5

        X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

        if feature == 'mfcc':
            X = librosa.feature.mfcc(y=audio, sr=rate, S=X.T, n_mfcc=40).T

        if feature == 'melsp_1':        
            X = librosa.feature.melspectrogram(y=audio, sr=rate, S=X.T, n_fft=400, hop_length=160).T
            X = librosa.power_to_db(X, ref=np.max)

        

    # apply global (featurewise) standardization to mean1, var0
        
        
        scaler = sklearn.preprocessing.StandardScaler()
        with np.load(os.path.join("models", 'scaler_' + feature + '.npz')) as data:
        #print('Scaler: mean - ', data['arr_0'], ' , scale - ', data['arr_0'])
            scaler.mean_ = data['arr_0']
            scaler.scale_ = data['arr_1']

        X = scaler.transform(X)


    # cut to input shape length (500 frames x 201 STFT bins)
        #X = X[:500, :]

    # apply l2 normalization
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)

        #if labels_f[k] < 5 and labels_f[k] >= 0:


        for i in range(int((len(audio)/rate) // sec) + 1):

                #LABELS[k, labels_f[k]] = 1
                #LABELS[len(DATA), labels_f[k]] = 1  # labels_f[k]-1 - для моделей с классами 1-4
            DATA.append(X[int(100*sec)*(i):int(100*sec)*(i+1), :])

        
        

    # add sample dimension (если нужно)
    #X = X[np.newaxis, ...]
    #if len(model.input_shape) == 4:
    #    X = X[:, np.newaxis, ...]

        DATA = np.asarray(DATA)
    #LABELS = np.asarray(LABELS)

    ######## 
    #scaler = sklearn.preprocessing.StandardScaler()


        print(DATA.shape)
        print(DATA[0].shape)
    #print(DATA[0].shape)

        with open('F:/amicorpus_2/original_pickle/' + names_f[k] + '.pickle', 'wb') as f:
        #pickle.dump(DATA[:half][:][:], f)
            pickle.dump(DATA, f)

    #with open('data/data_2.pickle', 'wb') as f:
        #pickle.dump(DATA[half:][:][:], f)

    #with open('data/labels_0-4_' + str(sec) + 'sec.pickle', 'wb') as f:
    #    pickle.dump(LABELS, f)


    print('Data prepare complete! Seconds: ' + str(sec) + ' Features: ' + feature)

if __name__ == '__main__':

    audio_path = 'F:/amicorpus_2/original'



    path_f = []
    names_f = []
    for d, dirs, files in os.walk(audio_path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            path_f.append(path) # добавление адреса в список
            names_f.append(f.split('.')[0])

    #sec = [2, 1.5, 1, 0.5]
    sec = [1]
    #sec = [5, 4, 3, 2]
    #features = ['mfcc', 'melsp_1']
    features = ['stft']
    for s in sec:
        for feat in features:
            prepare_data(path_f[:2], names_f, s, feat)
