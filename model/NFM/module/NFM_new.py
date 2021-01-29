# -*- coding:utf-8 -*-
 
"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of NFM


Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics
    Xiangnan He,School of Computing,National University of Singapore,Singapore 117417,dcshex@nus.edu.sg
    Tat-Seng Chua,School of Computing,National University of Singapore,Singapore 117417,dcscts@nus.edu.sg

"""

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utility.DataLoader import DataLoader
from utility.Trainer import Trainer
from .BaseModule import BaseModule
"""
    网络结构部分
"""

class NFM(BaseModule):

    def __init__(self,field_size, feature_sizes, embedding_size = 4, is_shallow_dropout = True, dropout_shallow = [0.5],
                 h_depth = 2, deep_layers = [32, 32], is_deep_dropout = True, dropout_deep=[0.0, 0.5, 0.5],
                 deep_layers_activation = 'relu', n_epochs = 64, batch_size = 256, learning_rate = 0.003,
                 optimizer_type = 'adam', is_batch_norm = False, verbose = 10, random_seed = 970801, weight_decay = 0.0,
                 use_fm = True, use_ffm = False, interation_type = True,loss_type = 'logloss', eval_metric = roc_auc_score,
                 use_cuda = True, n_class = 1, greater_is_better = True,save_path = None
                 ):
        super(NFM, self).__init__()
        self.field_size = field_size  # size of the feature fields
        self.feature_sizes = feature_sizes # a field_size-dim array, sizes of the feature dictionary
        self.embedding_size = embedding_size # size of the feature embedding
        self.is_shallow_dropout = is_shallow_dropout # bool, shallow part(fm or ffm part) uses dropout or not?
        self.dropout_shallow = dropout_shallow # an array of the size of 1, example:[0.5], the element is for the-first order part
        self.h_depth = h_depth # deep network's hidden layers' depth
        self.deep_layers = deep_layers # a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
        self.is_deep_dropout = is_deep_dropout # bool, deep part uses dropout or not?
        self.dropout_deep = dropout_deep # an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
        self.deep_layers_activation = deep_layers_activation # relu or sigmoid etc
        self.n_epochs = n_epochs # epochs
        self.batch_size = batch_size # batch_size
        self.learning_rate = learning_rate # learning_rate
        self.optimizer_type = optimizer_type # optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
        self.is_batch_norm = is_batch_norm # bool,  use batch_norm or not ?
        self.verbose = verbose # verbose
        self.weight_decay = weight_decay # weight decay (L2 penalty)
        self.random_seed = random_seed # random_seed=970801 someone's birthday, my lukcy number
        self.use_fm = use_fm # bool
        self.use_ffm = use_ffm # bool
        # bool, When it's true, the element-wise product of the fm or ffm embeddings will be added together,
        # otherwise, the element-wise prodcut of embeddings will be concatenated.
        self.interation_type = interation_type
        self.loss_type = loss_type # "logloss", only
        self.eval_metric = eval_metric # roc_auc_score
        self.use_cuda = use_cuda # bool use gpu or cpu?
        self.n_class = n_class # number of classes. is bounded to 1
        self.greater_is_better = greater_is_better # bool. Is the greater eval better?
        self.save_path = save_path
        torch.manual_seed(self.random_seed)

        self.best_iteration_ = None
        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is nfm(fm+nn layers)")
        elif self.use_ffm:
            print("The model is nffm(ffm+nn layers)")
        else:
            print("You have to choose more than one of (fm, ffm) models to use")
            exit(1)
        """
            bias
        """
        self.bias = torch.nn.Parameter(torch.randn(1))

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            # field_size 个embedding (feature_size,1)
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            # field_size 个embedding 独立层
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            print("Init ffm part succeed")

        """
            deep part
        """
        print("Init deep part")
        # 第一层
        if self.is_deep_dropout:
            self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
        if self.interation_type:
            self.linear_1 = nn.Linear(self.embedding_size, deep_layers[0])
        else:
            self.linear_1 = nn.Linear(int(self.field_size*(self.field_size-1)/2), deep_layers[0])
        if self.is_batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
        if self.is_deep_dropout:
            self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
        # 第二层以后
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

        print("Init deep part succeed")

        print ("Init succeed")

    def forward(self, data):
        Xi = data['Xi'] # batch, feature,1
        Xv = data['Xv'] # batch, feature

        """ fm part """
        if self.use_fm:
            # .t() 转置
            fm_first_order_emb_arr = [( torch.sum( emb(Xi[:,i,:]) ,1).t() * Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr,1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

            if self.interation_type:
                # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
                fm_second_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
                fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
                fm_sum_second_order_emb_square = fm_sum_second_order_emb*fm_sum_second_order_emb # (x+y)^2
                fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
                fm_second_order_emb_square_sum = sum(fm_second_order_emb_square) #x^2+y^2
                fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            else:
                fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                           enumerate(self.fm_second_order_embeddings)]
                fm_wij_arr = []
                for i in range(self.field_size):
                    for j in range(i + 1, self.field_size):
                        fm_wij_arr.append(fm_second_order_emb_arr[i] * fm_second_order_emb_arr[j])
        """
            ffm part
        """
        if self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr,1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:,i,:]), 1).t() * Xv[:,i]).t() for emb in  f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    ffm_wij_arr.append(ffm_second_order_emb_arr[i][j]*ffm_second_order_emb_arr[j][i])
            ffm_second_order = sum(ffm_wij_arr)

        """
            deep part
        """
        if self.use_fm and self.interation_type:
            deep_emb = fm_second_order
        elif self.use_ffm and self.interation_type:
            deep_emb = ffm_second_order
        elif self.use_fm:
            deep_emb = torch.cat([torch.sum(fm_wij,1).view([-1,1]) for fm_wij in fm_wij_arr], 1)
        else:
            deep_emb = torch.cat([torch.sum(ffm_wij,1).view([-1,1]) for ffm_wij in ffm_wij_arr],1)

        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = torch.tanh
        else:
            activation = F.relu

        if self.is_deep_dropout:
            deep_emb = self.linear_0_dropout(deep_emb)
        x_deep = self.linear_1(deep_emb)
        if self.is_batch_norm:
            x_deep = self.batch_norm_1(x_deep)
        x_deep = activation(x_deep)
        if self.is_deep_dropout:
            x_deep = self.linear_1_dropout(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)

        """
            sum
        """
        if self.use_fm:
            total_sum = self.bias+ torch.sum(fm_first_order,1) + torch.sum(x_deep,1)
        elif self.use_ffm:
            total_sum = self.bias + torch.sum(ffm_first_order, 1) + torch.sum(x_deep, 1)
        return total_sum


    def fit(self, Xi_train, Xv_train,  y_train,eval_set=None,
            early_stopping_rounds=None, verbose=None,eval_metric=['f1']):

        trainDataLoader = DataLoader(Xi_train,Xv_train,y_train,batch_size= self.batch_size)
        validDataLoader_list = None
        if eval_set != None:
            validDataLoader_list = []
            for valid_data in eval_set:
                Xi_valid = valid_data[0]
                Xv_valid = valid_data[1]
                y_valid = valid_data[2]
                validDataLoader_list.append(DataLoader(Xi_valid, Xv_valid, y_valid, batch_size=self.batch_size))

        if self.verbose:
            print("pre_process data finished")
        """
            train model
        """
        trainer = Trainer(model=self,
                         train_data_loader=trainDataLoader,
                         eval_set =validDataLoader_list,
                         train_times=self.n_epochs,
                         metric = eval_metric,
                         opt_method=self.optimizer_type,
                         alpha= self.learning_rate,
                         weight_decay=self.weight_decay,
                         lr_decay=0,
                         save_steps=100,
                         early_stopping_rounds = early_stopping_rounds,
                         verbose = verbose,
                         use_gpu=False,
                         checkpoint_dir=self.save_path)
        trainer.run()


    def predict(self, Xi=None, Xv=None ,data=None, num_iteration=-1):
        if num_iteration != -1:
            self.load_checkpoint(self.save_path+f'model_iteration_{num_iteration}.pkl')

        if data == None:
            Xi = np.array(Xi).reshape((-1, self.field_size, 1))
            Xi = Variable(torch.LongTensor(Xi))
            Xv = Variable(torch.FloatTensor(Xv))
            data = {"Xi": Xi, "Xv": Xv}
        return (self.predict_proba(data=data) > 0.5)

    def predict_proba(self, Xi=None, Xv=None,data=None, num_iteration=-1):
        if num_iteration!=-1:
            self.load_checkpoint(self.save_path+f'model_iteration_{num_iteration}.pkl')
        if data == None:
            Xi = np.array(Xi).reshape((-1, self.field_size, 1))
            Xi = Variable(torch.LongTensor(Xi))
            Xv = Variable(torch.FloatTensor(Xv))
            data = {"Xi": Xi, "Xv": Xv}
        if self.use_cuda and torch.cuda.is_available():
            data['Xi'], data['Xv'] = data['Xi'].cuda(), data['Xv'].cuda()

        model = self.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(data)).cpu()
        return pred.data.numpy()

    def get_hidden_layer(self,Xi,Xv):
        with torch.no_grad():
            Xi = np.array(Xi).reshape((-1, self.field_size, 1))
            Xi = Variable(torch.LongTensor(Xi))
            Xv = Variable(torch.FloatTensor(Xv))

            if self.use_fm:
                fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                          enumerate(self.fm_first_order_embeddings)]
                fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

                if self.interation_type:
                    # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
                    fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                               enumerate(self.fm_second_order_embeddings)]
                    fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
                    fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
                    fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
                    fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
                    fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
                else:
                    fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                               enumerate(self.fm_second_order_embeddings)]
                    fm_wij_arr = []
                    for i in range(self.field_size):
                        for j in range(i + 1, self.field_size):
                            fm_wij_arr.append(fm_second_order_emb_arr[i] * fm_second_order_emb_arr[j])
            """
                ffm part
            """
            if self.use_ffm:
                ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                           enumerate(self.ffm_first_order_embeddings)]
                ffm_first_order = torch.cat(ffm_first_order_emb_arr, 1)
                ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs] for
                                            i, f_embs in enumerate(self.ffm_second_order_embeddings)]
                ffm_wij_arr = []
                for i in range(self.field_size):
                    for j in range(i + 1, self.field_size):
                        ffm_wij_arr.append(ffm_second_order_emb_arr[i][j] * ffm_second_order_emb_arr[j][i])
                ffm_second_order = sum(ffm_wij_arr)

            """
                deep part
            """
            if self.use_fm and self.interation_type:
                deep_emb = fm_second_order
            elif self.use_ffm and self.interation_type:
                deep_emb = ffm_second_order
            elif self.use_fm:
                deep_emb = torch.cat([torch.sum(fm_wij, 1).view([-1, 1]) for fm_wij in fm_wij_arr], 1)
            else:
                deep_emb = torch.cat([torch.sum(ffm_wij, 1).view([-1, 1]) for ffm_wij in ffm_wij_arr], 1)

            # if self.deep_layers_activation == 'sigmoid':
            #     # activation = F.sigmoid
            #     activation = torch.sigmoid
            # elif self.deep_layers_activation == 'tanh':
            #     activation = F.tanh
            # else:
            #     activation = F.relu
            #
            # x_deep = self.linear_1(deep_emb)
            # if self.is_batch_norm:
            #     x_deep = self.batch_norm_1(x_deep)
            # x_deep = activation(x_deep)
            # for i in range(1, len(self.deep_layers)):
            #     x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
            #     if self.is_batch_norm:
            #         x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            #     x_deep = activation(x_deep)

        return deep_emb.data.numpy()

    def print_embedding_prod(self,Xi,Xv):
        if not self.use_fm:
            print ("Error! Only print fm model!")
            return
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                   enumerate(self.fm_second_order_embeddings)]
        total_prod = fm_second_order_emb_arr[0] + 1.0
        for emb in fm_second_order_emb_arr[1:]:
            total_prod = total_prod * (emb + 1.0)
        print ("max:", torch.max(total_prod))
        print ("min", torch.min(total_prod))

"""
    test part
"""
