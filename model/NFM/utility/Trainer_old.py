# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from .helper import optimizer_factory,metric_name2func ,early_stopping,ensureDir

class Trainer(object):

    def __init__(self,
                 model=None,
                 train_data_loader=None,
                 valid_data_loader =None,
                 train_times=1000,
                 metric = ['auc'],
                 opt_method="Adam",
                 optimizer = None,
                 alpha=0.5,
                 weight_decay=0.0,
                 lr_decay=0,
                 save_steps=100,
                 early_stopping_rounds = None,
                 verbose =None,
                 use_gpu=False,
                 checkpoint_dir=None):

        self.model = model
        self.train_data_loader = train_data_loader  # 一个对象，训练使用的数据集 ，产生一个 epoch 的数据，即 nbatches 个batches的数据
        self.valid_data_loader = valid_data_loader  #
        self.train_times = train_times  # 训练轮数
        self.metrics = metric
        self.metric_funcs = []
        for mc in self.metrics:
            self.metric_funcs.append(metric_name2func(mc))
        self.criterion  = F.binary_cross_entropy_with_logits

        self.opt_method = opt_method  # 优化器 的名字 字符串
        self.optimizer = optimizer  # 相对应的 优化器的对象
        self.alpha = alpha  # 学习率
        self.lr_decay = lr_decay  # 学习率衰减
        self.weight_decay = weight_decay  #

        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.save_steps = save_steps  #
        self.scheduler =None

        self.use_gpu = use_gpu  # 是否使用gpu
        self.checkpoint_dir = checkpoint_dir  # 检查点保存地址
        if self.checkpoint_dir!=None:
            ensureDir(checkpoint_dir)

    # 训练一个batch 数据的过程
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        outputs = self.model(data)  # size is 1
        loss = self.criterion(outputs, data['label'])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def get_metric_scores(self,data):
        outputs = self.model.predict_proba(data=data)
        result = {}
        for metric_name,metric_func in zip(self.metrics,self.metric_funcs):
            result[metric_name] = metric_func(data['label'].data.numpy(),outputs)
        return result

    # 整个训练过程的 配置，以及调用 一个训练batcch方法的框架
    def run(self):
        if self.use_gpu:
            self.model.cuda()
        # 配置优化器
        self.optimizer = optimizer_factory(self.model,
                                            opt_method=self.opt_method,
                                            alpha=self.alpha,
                                            lr_decay=self.lr_decay,
                                            weight_decay=self.weight_decay)
        print("Finish initializing...")
        self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, pct_start=0.05, div_factor=1.5e3,
                                              max_lr=1e-2, epochs=self.train_times, steps_per_epoch=len(self.train_data_loader))
        best_value = 0
        stopping_step = 0
        # 在训练模型
        for epoch in range(self.train_times):  # 迭代 train_times 次 epoch
            self.model = self.model.train()
            res = 0.0
            for data in self.train_data_loader:  # 迭代 nbatches 次 batch
                loss = self.train_one_step(data)  # size is 1
                res += loss
            res = res / self.train_data_loader.nbathes

            with torch.no_grad():
                self.model = self.model.eval()
                train_score = self.get_metric_scores(self.train_data_loader.get_whole_data())
                if self.valid_data_loader!=None:
                    valid_score = self.get_metric_scores(self.valid_data_loader.get_whole_data())
            if self.verbose!=None and epoch%self.verbose==0:
                self.verbose_print(epoch,res,train_score,valid_score)

            if self.save_steps!=None and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                ensureDir(self.checkpoint_dir)
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

            if self.early_stopping_rounds!=None:
                early_stopping_score = train_score[self.metrics[0]] if self.valid_data_loader == None else valid_score[self.metrics[0]]
                best_value, stopping_step, should_stop = early_stopping(early_stopping_score,
                                                                        best_value,
                                                                        stopping_step,
                                                                        flag_step=self.early_stopping_rounds)
                if stopping_step == 0 and self.checkpoint_dir!=None:
                    self.model.best_iteration_ = epoch
                    self.model.save_checkpoint(self.checkpoint_dir+f'model_iteration_{self.model.best_iteration_}.pkl')
                if should_stop:
                    print(f"Training until validation scores don't improve for 20 rounds, and best_value is",best_value)

                    break


    def verbose_print(self,epoch,loss,train_score,valid_score):
        msg = f"[{epoch}]   train loss is {loss}"
        train_msg = '    train score  '
        for key in train_score:
            train_msg += f"{key}:{train_score[key]}  "
        valid_msg = '    valid score  '
        for key in valid_score:
            valid_msg += f"{key}:{valid_score[key]}  "
        print(msg)
        print(train_msg)
        print(valid_msg)

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir