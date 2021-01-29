import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

class DataLoader(object):
    def __init__(self,Xi_data = None, Xv_data = None, label = None,addition_feature=None, batch_size=2 ):

        self.Xi_data = np.array(Xi_data).reshape((-1,Xi_data.shape[1],1))
        self.Xv_data = np.array(Xv_data)

        if addition_feature is not None:
            self.addition_feature = np.array(addition_feature)
        else:
            self.addition_feature = None

        self.label = np.array(label)
        self.batch_size = batch_size
        self.data_size = Xi_data.shape[0]

        self.nbathes = int(self.data_size/batch_size)
        if self.data_size%batch_size:
            self.nbathes+=1


    def __len__(self):
        return  (self.nbathes)

    def __getitem__(self, idx):
        if idx >= self.nbathes:
            raise IndexError(idx)
        offset = idx * self.batch_size
        end = min(self.data_size, offset + self.batch_size)
        data = {
            'Xi':Variable(torch.LongTensor(self.Xi_data[offset:end])),
            'Xv':Variable(torch.FloatTensor(self.Xv_data[offset:end])),
            'label':Variable(torch.FloatTensor(self.label[offset:end])),

        }
        if self.addition_feature is not None:
            data['addition_feature'] = Variable(torch.FloatTensor(self.addition_feature[offset:end]))
        else:
            data['addition_feature'] = None
        return data

    def get_whole_data(self):
        data = {
            'Xi': Variable(torch.LongTensor(self.Xi_data)),
            'Xv': Variable(torch.FloatTensor(self.Xv_data)),
            'label': Variable(torch.FloatTensor(self.label))
        }
        if self.addition_feature is not None:
            data['addition_feature'] = Variable(torch.FloatTensor(self.addition_feature))
        else:
            data['addition_feature'] = None
        return data