'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
import torch
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,recall_score

def my_f1_score(y_true,y_pre):
    return f1_score(y_true,(y_pre>0.5))
def my_accuracy_score(y_true,y_pre):
    return accuracy_score(y_true,(y_pre>0.5))
def my_recall_score(y_true,y_pre):
    return recall_score(y_true,(y_pre>0.5))

def metric_name2func(metric_name):
    if metric_name == 'auc' or metric_name == 'AUC':
        return roc_auc_score
    elif metric_name == 'f1' or metric_name == 'F1':
        return my_f1_score
    elif metric_name == 'accuracy' or metric_name == 'acc':
        return my_accuracy_score
    elif metric_name == 'recall':
        return my_recall_score
# 配置优化器
def optimizer_factory(model,opt_method='sgd',alpha=0.001,lr_decay=0.0,weight_decay=0.0):
    if opt_method == "Adagrad" or opt_method == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=alpha,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "Adadelta" or opt_method == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "Adam" or opt_method == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
            )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,  # L2正则化
        )
    return optimizer



def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

