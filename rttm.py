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


#file_csv = 'outputs\\AMIsilence\\' + name + '.no_silence.csv'

file_rttm = 'outputs\\logunov\\logunov\\speech_regions_predicted_vad_markup_bad_repaired\\EN2001a.segments.rttm'

data = []
with open(file_rttm, encoding='utf-8') as r_file:
    for line in r_file:
        data = line.split(' ')
        print(float(data[2]), float(data[3]))