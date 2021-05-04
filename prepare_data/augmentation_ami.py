import numpy as np
import soundfile as sf
import librosa
import pickle
import pandas as pd

import random #(??)

def augment_file(sec, path, name):

    # 1. считать разметку и файл
    # 2. выбрать рандомное место и считать метки в течении Х фрагментов/секунд
    # 3. если встречается 0, 3, 4: заново п.2
    # 4. повторить п.2
    # 5. сложить фрагменты и разметку
    # 6. посмотреть баланс классов
    # 7. если 1 намного больше -> повторить п.2-6, пока не будет баланса
    # 8. сохранить новые разметку и файл









def main():

    paths = []
    names = []

    for d, dirs, files in os.walk('F:\\amicorpus_1'): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            name = f.split('.')[0]

            paths.append(path) # добавление адреса в список
            names.append(name)


    sec = 10 # или сегментов

    for i in range(len(paths)):    
        augment_file(sec, paths[i], names[i])


if __name__ == '__main__':
    main()
