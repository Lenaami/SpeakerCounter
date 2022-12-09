import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import random
from scipy.io import wavfile
from utils_.data_list import get_data_list
import os

def get_rand_start(sec, data):
    tries = 0
    countinue = True
    sum_sec = 0

    search = [[0, 3, 4], [0, 4], [3, 4]]

    area = search[0]
    while countinue:
        tries += 1
        countinue = False
        sum_sec = 0
        rand_start = random.uniform(0, data[-1, 0] + data[-1, 1] - sec)

        for start, dur, count in data:
            if (start > rand_start or (start < rand_start and start + dur > rand_start)) and sum_sec < sec:
                sum_sec += dur
                if count in area: #[0, 3, 4]:
                    countinue = True

        if countinue == False:
            break
        if tries == 100:
            #print('Area 1')
            area = [1]
        if tries == 200:
            #print('Area 2')
            area = [2]
        if tries == 300:
            print('A lot of tries (rand)')
            break

    # print('Tries: ', tries)
    # print('Rand_start: ', rand_start)
    return rand_start


def check_balance(data, balance=False, print_stat=False):  # Формат start, duration, count
    duration = [[], [], [], [], []]
    for st, dur, cnt in data:
        sp = int(cnt)
        if sp > 4: sp = 4
        duration[sp].append(dur)

    # Не учитываем нулевой класс
    sum_dur = 0
    for i in range(1, len(duration)):
        sum_dur += np.sum(duration[i])


    avg_dur = sum_dur / (len(duration) - 1) # Средняя длительность в секундах (классы 1-4)
    sum_dur += np.sum(duration[0])

    needBalance = False
    count_false = 0
    perc = []
    if print_stat: print('Процент класса 0: ', np.sum(duration[0]) / sum_dur * 100)
    perc.append(np.sum(duration[0]) / sum_dur * 100)
    for i in range(1, len(duration)):
        perc.append(np.sum(duration[i]) / sum_dur * 100)
        if np.abs(np.sum(duration[i]) - avg_dur) > avg_dur * 0.2:  # разница была не больше 20% (+- 4 % от датасета)
            if print_stat: print('Процент класса ' + str(i) + ': ', np.sum(duration[i]) / sum_dur * 100)
            #needBalance = True
            count_false += 1
        else:
            if print_stat: print('(OK) Процент класса ' + str(i) + ': ', np.sum(duration[i]) / sum_dur * 100)
    if print_stat: print('-----')

    #print(sum_dur)
    #print(perc)

    '''
    perc = []
    for i in range(1, len(duration)):
        print('Процент класса ' + str(i) + ': ', np.sum(duration[i]) / (sum_dur + np.sum(duration[0])) * 100)
        perc.append(np.sum(duration[i]) / (sum_dur + np.sum(duration[0])) * 100)
    print('-----')
    '''

    if count_false > 2:
        needBalance = True

    if balance:
        return perc
    else:
        return needBalance


def augment_file(sec, path, name, iterations):
    # 1. считать разметку и файл
    # 2. выбрать случайное место и считать метки в течении Х фрагментов/секунд
    # 3. если встречается 0, 3, 4: заново п.2
    # 4. повторить п.2
    # 5. сложить фрагменты и разметку
    # 6. посмотреть баланс классов
    # 7. если 1 намного больше -> повторить п.2-6, пока не будет баланса
    # 8. сохранить новые разметку и файл

    # t_start = time.time()

    file_csv = './data/AMI/Count_new/' + name + '.count.csv'  # путь к файлам
    #file_csv = './data/AMI/Count_aug_3/' + name + '.count.csv'  # путь к файлам

    DATA = pd.read_csv(file_csv, delimiter=',')  # колонки - start, duration, count или (start, end, duration, count)
    DATA['duration'] = DATA['duration'].round(3)
    data = DATA.values

    print('Файл: ' + name)

    check_balance(data, print_stat=True)

    parts = []
    h = 0

    for i in tqdm(range(iterations)):
    #while check_balance(data) or h < iterations:

        h += 1
        try:
        #for h in range(1):

            rand_parts = np.asarray(parts)
            countinue = True
            tries = 0
            while countinue:

                rand_start_1 = round(get_rand_start(sec, data), 3) # что копируется

                rand_start_2 = round(get_rand_start(sec, data), 3) # куда копируется

                while abs(rand_start_2 - rand_start_1) < sec:
                    rand_start_2 = round(get_rand_start(sec, data), 3)

                if rand_parts.shape[0] == 0:
                    countinue = False
                else:
                    if np.min(np.abs(rand_start_1 - rand_parts[:, 0])) < sec or np.min(np.abs(rand_start_2 - rand_parts[:, 1])) < sec:
                        countinue = True
                    else:
                        countinue = False
                tries += 1
                if tries == 100:
                    print('A lot of tries')
                    break


            sum_sec = 0
            data_1 = []
            data_1.append([0.0, rand_start_2, 0])  # Первая строчка сразу с нужной длительностью
            for start, dur, count in data:
                if (start > rand_start_1 or (start < rand_start_1 and start + dur > rand_start_1)) and round(sum_sec, 3) < sec:
                    if dur - (rand_start_1 - start) > sec and len(data_1) == 1:  # попал целый кусочек записи
                        data_1.append([rand_start_1, sec, count])
                        sum_sec += sec
                    elif len(data_1) == 1:  # если первый элемент
                        data_1.append([rand_start_1, round(dur - (rand_start_1 - start), 3), count])
                        sum_sec += round(dur - (rand_start_1 - start), 3)
                    elif rand_start_1 + sec < start + dur:  # если последний элемент
                        data_1.append([start, round(rand_start_1 + sec - start, 3), count])
                        sum_sec += round(rand_start_1 + sec - start, 3)
                    else:  # все остальные
                        data_1.append([start, dur, count])
                        sum_sec += dur
            data_1.append(
                [rand_start_1 + sec, data[-1, 0] + data[-1, 1] - (rand_start_2 + sec), 0])  # Также с нужной длительностью
            # print(data_1)

            for i in range(len(data_1) - 1):  # выравниваем время
                if rand_start_1 < rand_start_2:
                    data_1[i + 1][0] += rand_start_2 - rand_start_1
                else:
                    data_1[i + 1][0] -= rand_start_1 - rand_start_2
                    # print(data_1)

            data_2 = data.copy()
            data_2[:, 1] = data_2[:, 0] + data_2[:, 1]
            data_1 = np.asarray(data_1)
            data_1[:, 1] = data_1[:, 0] + data_1[:, 1]

            # Заглушка (кривое вычисление конца)
            data_1[-1, 1] = data[-1, 0] + data[-1, 1]

            # print(data_1)

            DATA_1 = pd.DataFrame(data_1, columns=['start', 'end', 'count'])
            DATA_1['start'] = DATA_1['start'].round(3)
            DATA_1['end'] = DATA_1['end'].round(3)
            DATA_2 = pd.DataFrame(data_2, columns=['start', 'end', 'count'])
            DATA_2['start'] = DATA_2['start'].round(3)
            DATA_2['end'] = DATA_2['end'].round(3)

            # print(DATA_1)

            time = list(DATA_1['start'].values)
            time.extend(list(DATA_1['end'].values))
            # print(time)
            time.extend(list(DATA_2['start'].values))
            time.extend(list(DATA_2['end'].values))
            time = sorted(list(set(time)))

            # print(time)
            # Создание новой разметки
            speakers = [0] * (len(time) - 1)
            data = []

            idx_st = time.index(rand_start_2)
            idx_ed = time.index(round(rand_start_2 + sec, 3))

            for i in range(len(time) - 1):

                if i < idx_st - 1 or i > idx_ed + 1:
                    k, _ = np.where(data_2 == time[i])
                    if k.shape[0] > 1: k = k[1]
                    # print(data_2[k], tuple(data_2[k]), k, time[i], time[i+1])
                    st, ed, pers = data_2[k]
                    data.append([st, ed - st, int(pers)])
                else:
                    for st, ed, pers in data_1:
                        if (time[i + 1] - st) > 0 and (ed - time[i]) > 0:
                            speakers[i] += pers
                    for st, ed, pers in data_2:
                        if (time[i + 1] - st) > 0 and (ed - time[i]) > 0:
                            speakers[i] += pers
                    data.append([time[i], time[i + 1] - time[i], int(speakers[i])])

            data = np.asarray(data)

            parts.append([rand_start_1, rand_start_2])
        except Exception:
            print('Skip iteration')

    # t_stop = time.time()
    # print("Augment time : " + str(t_stop - t_start))

    # сохранить новый файл csv

    print('Количество итераций: ', h)

    COUNT = pd.DataFrame(data, columns=['start', 'duration', 'count'])
    COUNT.to_csv('./data/AMI/Count_aug_mh/' + name + '.count.csv', index=False)

    # запись и сохранение аудио
    fs, audio = wavfile.read(path)
    au = audio.copy()
    duration = int(sec * fs)
    for rand_start_1, rand_start_2 in parts:
        au[int(fs * rand_start_2):int(fs * (rand_start_2)) + duration] += au[int(fs * rand_start_1):int(fs * rand_start_1) + duration]

    wavfile.write('./data/AMI/augmented_mh/' + name + '.Mix-Headset.wav', fs, au)

    #print(parts)

    check_balance(data, print_stat=True)


def main():

    with open('./yaml/ami_sets_mini20.yaml') as f:
        ami_sets = yaml.load(f, Loader=yaml.FullLoader)
    names_au = ami_sets['train']

    #names_au = ['ES2005c', 'ES2002d'] # Заменены с 'ES2005a', 'ES2002c'

    names, paths = get_data_list(names_au, 'Mix-Headset', '/srv/data/AMI/amicorpus/')

    #names_1, paths_1 = [], []

    #files = os.listdir('./data/AMI/Count_new')
    DATA = pd.DataFrame({'start': [], 'duration': [], 'count': []})

    '''
    for i in range(len(names)):
        file_csv = './data/AMI/Count_aug_4/' + names[i] + '.count.csv'# путь к файлам
        data = pd.read_csv(file_csv, delimiter=',') # колонки - start, duration, count или (start, end, duration, count)

        #print('File: ' + names[i])
        #print(data)
        #perc = check_balance(data.values, balance=True)
        #print(perc)
        #if (perc[3] < 10 and perc[4] < 10) or perc[3]==0.0 :
        #    names_1.append(names[i])
        #    paths_1.append(paths[i])

        #names_1.append(names[i])
        #paths_1.append('./data/AMI/augmented_1/' + names[i] + '.Array1-01.wav')

        DATA = pd.concat([DATA, data])

    data = DATA.values
    check_balance(data, print_stat=True)

    '''

    sec = 15  # секунд (или сегментов)
    iterations = 15# максимальное количество итераций

    '''
    names_1, paths_1 = [], []
    for_fix = ['TS3008a']
    for f in for_fix:
        names_1.append(f)
        paths_1.append('./data/AMI/augmented_3/' + f + '.Array1-01.wav')
        #paths_1.append('/srv/data/AMI/amicorpus/' + f + '/audio/' + f + '.Array1-01.wav')

    for i in range(len(paths_1)):
        augment_file(sec, paths_1[i], names_1[i], iterations)
    '''

    for i in range(len(paths)):
        augment_file(sec, paths[i], names[i], iterations)
    #augment_file(sec, paths[0], names[0], iterations)

if __name__ == '__main__':
    main()
