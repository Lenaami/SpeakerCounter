# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

classes = {
    4: '1-4',
    5: '0-4'
}


class model_LSTM(nn.Module):

    def __init__(self, n_classes, sec, features, **kwargs):
        super().__init__()

        units_reshape_1 = {
            500: 53,
            400: 42,
            300: 31,
            200: 20,
            150: 15,
            100: 9,
            50: 3
        }

        units_reshape_2 = {
            'stft': 1280,
            'melsp': 768,
            'mfcc': 192
        }

        input_features = {
            'stft': 201,
            'melsp': 128,
            'mfcc': 40
        }

        units_dense = {
            500: 2120,
            400: 1680,
            300: 1240,
            200: 800,
            150: 600,
            100: 360,
            50: 120
        }

        input_length = int(100 * sec)
        n_input_features = input_features[features]  # 201 - stft, 128 - melsp, 40 - mfcc
        n_units_reshape_1 = units_reshape_1[input_length]
        n_units_reshape_2 = units_reshape_2[features]
        n_units_dense_0 = units_dense[input_length]
        n_units_dense = n_classes

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 3), stride=3)
        self.mp2 = nn.MaxPool2d(kernel_size=(3, 2), stride=3)
        self.mp3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d(0.3)
        self.lstm = nn.LSTM(n_units_reshape_2, 40, num_layers=1, batch_first=True, bidirectional=True)
        self.flat = nn.Flatten()
        #self.linn = nn.Linear(5760, n_units_dense)
        self.lin = nn.Linear(n_units_dense_0, n_units_dense)


    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.mp1(x)
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = self.mp2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        #h0 = Variable(torch.randn(1, 125, 40)) # (num_layers, batch, hidden_size)
        #c0 = Variable(torch.randn(1, 125, 40))
        hidden = (torch.randn(2, x.shape[0], 40),#.cuda(),
                  torch.randn(2, x.shape[0], 40))#.cuda())
        #x, _ = self.lstm(x, (h0, c0)) #torch.tanh
        x, _ = self.lstm(x, hidden)
        #x, _ = self.lstm(x)
        x = self.mp3(x)
        x = self.flat(x)
        x = self.lin(x)
        #x = F.softmax(x, dim=1)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x, dim=1)
        return x


class model_CNN(nn.Module):

    def __init__(self, n_classes, sec, features, **kwargs):
        super().__init__()

        units_reshape_1 = {
            500: 53,
            400: 42,
            300: 31,
            200: 20,
            150: 15,
            100: 9,
            50: 3
        }

        units_reshape_2 = {
            'stft': 1280,
            'melsp': 768,
            'mfcc': 192
        }

        input_features = {
            'stft': 201,
            'melsp': 128,
            'mfcc': 40
        }

        units_dense_2 = {
            'stft': 17,
            'melsp': 8,
            'mfcc': 2
        }

        units_dense_1 = {
            500: 7,
            400: 5,
            300: 4,
            200: 2,
            150: 1,
            100: 1,
            50: 1
        }

        units_dense_1_mfcc = {
            500: 8,
            400: 6,
            300: 5,
            200: 3,
            150: 2,
            100: 1,
            50: 1
        }

        input_length = int(100 * sec)
        n_input_features = input_features[features]  # 201 - stft, 128 - melsp, 40 - mfcc
        n_units_reshape_1 = units_reshape_1[input_length]
        n_units_reshape_2 = units_reshape_2[features]
        n_units_dense_0 = 64 * units_dense_1[input_length] * units_dense_2[features]
        if features == 'mfcc':
            n_units_dense_0 = 64 * units_dense_1_mfcc[input_length] * units_dense_2[features]
        n_units_dense = n_classes


        n_filters_L1 = 16  # 32
        n_filters_L2 = 32  # 64
        n_filters_L3 = 64  # 128
        n_pool_sz_L1 = 1
        n_pool_sz_L2 = 2
        n_pool_sz_L3 = (1, 1)
        n_kernel_sz_L1 = 6
        n_kernel_sz_L2 = 4
        n_kernel_sz_L3 = 2
        if input_length == 50: n_kernel_sz_L3 = (1, 2) # было (1, 2)
        if features == 'mfcc': n_kernel_sz_L3 = (1, 1) # было (2, 1)
        n_strides_L1 = 1
        n_strides_L2 = 1
        n_strides_L3 = (2, 1)
        n_padding = (0, 0)
        if features == 'mfcc': n_padding = (0, 1) # отключено для stft mfcc , было (1, 0) if input_length == 50 and features == 'mfcc':
        n_units_dense_L1 = 512  # 1024
        n_units_dense_L2 = 256  # 512
        n_units_dense_L3 = 128  # 256
        f_dropout_prob_Lconv = 0.75
        f_dropout_prob_L1 = 0.1
        f_dropout_prob_L2 = 0.1
        f_dropout_prob_L3 = 0.5

        self.cnn1_1 = nn.Conv2d(in_channels=1, out_channels=n_filters_L1, kernel_size=n_kernel_sz_L1, stride=n_strides_L1)
        self.cnn1_2 = nn.Conv2d(in_channels=n_filters_L1, out_channels=n_filters_L1, kernel_size=n_kernel_sz_L1, stride=n_strides_L1)
        self.cnn1_3 = nn.Conv2d(in_channels=n_filters_L1, out_channels=n_filters_L1, kernel_size=n_kernel_sz_L1, stride=n_strides_L1)
        self.cnn2_1 = nn.Conv2d(in_channels=n_filters_L1, out_channels=n_filters_L2, kernel_size=n_kernel_sz_L2, stride=n_strides_L2, padding=n_padding)
        self.cnn2_2 = nn.Conv2d(in_channels=n_filters_L2, out_channels=n_filters_L2, kernel_size=n_kernel_sz_L2, stride=n_strides_L2, padding=n_padding)
        self.cnn2_3 = nn.Conv2d(in_channels=n_filters_L2, out_channels=n_filters_L2, kernel_size=n_kernel_sz_L2, stride=n_strides_L2, padding=n_padding)
        self.cnn3_1 = nn.Conv2d(in_channels=n_filters_L2, out_channels=n_filters_L3, kernel_size=n_kernel_sz_L3, stride=n_strides_L3)
        self.cnn3_2 = nn.Conv2d(in_channels=n_filters_L3, out_channels=n_filters_L3, kernel_size=n_kernel_sz_L3, stride=n_strides_L3)
        self.cnn3_3 = nn.Conv2d(in_channels=n_filters_L3, out_channels=n_filters_L3, kernel_size=n_kernel_sz_L3, stride=n_strides_L3)
        self.mp1 = nn.MaxPool2d(kernel_size=n_pool_sz_L1)
        self.mp2 = nn.MaxPool2d(kernel_size=n_pool_sz_L2)
        self.mp3 = nn.MaxPool2d(kernel_size=n_pool_sz_L3)
        self.drop_conv = nn.Dropout2d(f_dropout_prob_Lconv)
        self.bn = nn.BatchNorm2d(n_filters_L3)
        self.flat = nn.Flatten()
        self.drop_1 = nn.Dropout2d(f_dropout_prob_L1)
        self.drop_2 = nn.Dropout2d(f_dropout_prob_L2)
        self.drop_3 = nn.Dropout2d(f_dropout_prob_L3)
        self.lin_1 = nn.Linear(n_units_dense_0, n_units_dense_L1)
        self.lin_2 = nn.Linear(n_units_dense_L1, n_units_dense_L2)
        self.lin_3 = nn.Linear(n_units_dense_L2, n_units_dense_L3)
        self.lin_4 = nn.Linear(n_units_dense_L3, n_units_dense)

    def forward(self, x):
        x = F.relu(self.cnn1_1(x))
        x = self.mp1(x)
        x = F.relu(self.cnn1_2(x))
        x = self.mp1(x)
        x = F.relu(self.cnn1_3(x))
        x = self.mp1(x)

        x = F.relu(self.cnn2_1(x))
        x = self.mp2(x)
        x = F.relu(self.cnn2_2(x))
        x = self.mp2(x)
        x = F.relu(self.cnn2_3(x))
        x = self.mp2(x)

        x = F.relu(self.cnn3_1(x))
        x = self.mp3(x)
        x = F.relu(self.cnn3_2(x))
        x = self.mp3(x)
        x = F.relu(self.cnn3_3(x))
        x = self.mp3(x)

        x = self.drop_conv(x)
        x = self.bn(x)
        x = self.flat(x)

        #print(x.shape)

        x = torch.sigmoid(self.lin_1(x))
        x = self.drop_1(x)
        x = F.relu(self.lin_2(x))
        x = self.drop_2(x)
        x = F.relu(self.lin_3(x))
        x = self.drop_3(x)
        x = self.lin_4(x)
        #x = F.softmax(x, dim=1)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x, dim=1)
        return x


def load_model(**kwargs):  # model_arch, n_classes, sec, features, dataset, add_name

    model_name = create_model_name(**kwargs)

    if kwargs['model_arch'] == 'LSTM':
        model = model_LSTM(**kwargs)
    elif kwargs['model_arch'] == 'CNN':
        model = model_CNN(**kwargs)

    if not os.path.exists(kwargs['checkpoints_path'] + model_name + '/model.pt'):

        if not os.path.exists(kwargs['checkpoints_path'] + model_name):
            os.makedirs(kwargs['checkpoints_path'] + model_name)
    else:

        model.load_state_dict(torch.load(os.path.join(kwargs['checkpoints_path'] + model_name, 'model.pt')))

    return model, model_name


def create_model_name(model_arch, n_classes, sec, features, dataset, add_name, **kwargs):

    name = 'model_' + model_arch + '_' + classes[n_classes] + '_' + str(sec) + 'sec_' + features + '_' + dataset
    if add_name != '':
        name = name + '_' + add_name

    return name
