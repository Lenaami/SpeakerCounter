# coding=utf-8

import pandas as pd

def get_labels_AMI(name, sec, path_to_file_count):

    file_csv = path_to_file_count + name + '.count.csv' # путь к файлам

    DATA = pd.read_csv(file_csv, delimiter=',') # столбцы - start, duration, count или (start, end, duration, count)
    data = DATA.values
    data[:, 1] = data[:, 0] + data[:, 1] # переделываю в start, end, count

    count = []

    # разметка по секундам целого файла
    for i in range(int(data[-1, 1] // sec + 1)):
        persons = [0]
        for st, ed, pers in data:
            if ((i+1)*sec - st) > 0 and (ed - i*sec) > 0:
                persons.append(pers)
        count.append(max(persons))

    return count