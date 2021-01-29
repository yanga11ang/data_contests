import os
import numpy as np
import pandas as pd
import torch


class DataLoader(object):
    def __init__(self, file_path="../data/contest_data/train_data1.csv",
                 batch_size = 2,
                 rain_cols_cols=["Rain_sum"],
                 environment_cols=['T', 'w', 'wd'],
                 target_cols=["Qi"]):
        self.batch_size = batch_size
        self.n_bath = 0
        self.rain_cols = rain_cols_cols
        self.environment_cols = environment_cols
        self.target_cols = target_cols
        data = pd.read_csv(file_path)
        n_weeks = 4  # 可调
        n_input = n_weeks * 7 * 8
        n_out = 7 * 8
        self.to_supervised(data, n_input, n_out)
        print("Traindata is ready")

    def to_supervised(self, data, n_input, n_out, ):
        # encode 输入
        self.e_rain = []
        self.e_environment = []
        self.e_target = []
        # decode 输入
        self.d_rain = []
        self.d_environment = []
        self.d_target = []

        for in_start in range(len(data)):
            in_end = in_start + n_input
            out_end = in_end + n_out
            if out_end >= len(data):
                break
            self.e_rain.append(data[self.rain_cols].iloc[in_start:in_end].values)
            self.e_environment.append(data[self.environment_cols].iloc[in_start:in_end].values)
            self.e_target.append(data[self.target_cols].iloc[in_start:in_end].values)
            # decode 输入
            self.d_rain.append(data[self.rain_cols].iloc[in_end:out_end].values)
            self.d_environment.append(data[self.environment_cols].iloc[in_end:out_end].values)
            self.d_target.append(data[self.target_cols].iloc[in_end:out_end].values)

        self.e_rain = np.array(self.e_rain)  # (batch_size,n_input,feature_size)
        self.n_bath = self.e_rain.shape[0]/self.batch_size

        self.e_rain = np.split(self.e_rain, self.n_bath, 0)
        self.e_environment = np.split(np.array(self.e_environment), self.n_bath, 0)
        self.e_target = np.split(np.array(self.e_target), self.n_bath, 0)
        self.d_rain = np.split(np.array(self.d_rain), self.n_bath, 0)
        self.d_environment = np.split(np.array(self.d_environment), self.n_bath, 0)
        self.d_target = np.split(np.array(self.d_target), self.n_bath, 0)


    def __len__(self):
        return (self.e_target.shape[0])

    def __getitem__(self, idx):
        dct = {
            'e_rain': torch.tensor(self.e_rain[idx], dtype=torch.float),
            'e_environment': torch.tensor(self.e_environment[idx], dtype=torch.float),
            'e_target': torch.tensor(self.e_target[idx], dtype=torch.float),
            'd_rain': torch.tensor(self.d_rain[idx], dtype=torch.float),
            'd_environment': torch.tensor(self.d_environment[idx], dtype=torch.float),
            'd_target': torch.tensor(self.d_target[idx], dtype=torch.float),
        }
        return dct
