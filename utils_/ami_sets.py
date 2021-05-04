#- **SA** (TRAINING PART OF SEEN DATA): ES2002, ES2005, ES2006, ES2007, ES2008, ES2009, ES2010, ES2012, ES2013, ES2015, ES2016; IS1000, IS1001, IS1002 (no a), IS1003, IS1004, IS1005 (no d), IS1006, IS1007; TS3005, TS3008, TS3009, TS3010, TS3011, TS3012, EN2001, EN2003, EN2004a, EN2005a, EN2006, EN2009, IN1001, IN1002. IN1005, IN1007, IN1008, IN1009, IN1012, IN1013, IN1014, IN1016
#- **SB** (DEV PART OF SEEN DATA): ES2003, ES2011, IS1008, TS3004, TS3006, IB4001, IB4002, IB4003, IB4004, IB4010, IB4011
#- **SC** (UNSEEN DATA FOR EVALUATION): ES2004, ES2014, IS1009, TS3003, TS3007, EN2002

import os

train = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002' , 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012', 'EN2001', 'EN2003', 'EN2004', 'EN2005', 'EN2006', 'EN2009', 'IN1001', 'IN1002', 'IN1005', 'IN1007', 'IN1008', 'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016'] 
test = ['ES2003', 'ES2011', 'IS1008', 'TS3004', 'TS3006', 'IB4001', 'IB4002', 'IB4003', 'IB4004', 'IB4010', 'IB4011'] 
val = ['ES2004', 'ES2014', 'IS1009', 'TS3003', 'TS3007', 'EN2002'] 

def get_list_audio(set_files, path):


    # set - train, dev, eval
    # выдать пути к файлам

    # так же поменять в datagenerator имя на путь

    if set_files == 'train':
        names = train
    if set_files == 'test':
        names = test
    if set_files == 'val':
        names = val

    add_letter = ['', 'a', 'b', 'c', 'd', 'e'] # перебор различных комбинаций в названиях
    names_au = []

    for name in names:
        for l in add_letter:
            names_au.append[name + l]

    audio_names = {}

    for d, dirs, files in os.walk(path): # путь к файлам
        for f in files:
            path = os.path.join(d,f) # формирование адреса
            name = f.split('.')[0]
            if name in names_au: # если в сете есть такое имя
                audio_names[name] = path # добавление адреса в список

    return audio_names
