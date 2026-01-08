# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: dataset.py
# @time: 2025/12/25 09:34:13
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: data processing script, including load, normalization, etc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def Renewable_energy():
    df = pd.read_csv('./data/Baige_2019.csv',usecols=['PV','Tem','WS','Season'])
    pv = np.array(df['PV'])
    tem = np.array(df['Tem'])
    wind = np.array(df['WS'])
    season = np.array(df['Season'])
    PV_output = pv / 1000 * (1 - 0.35 * 0.01 * (tem - 25))

    # plt.figure(figsize=(12, 4))
    # plt.plot(pv, linewidth=1)
    # plt.savefig('./results/real_pv.svg', format="svg", bbox_inches='tight', dpi=300)

    # restruct (8760, ) --> (365, 24) / (365, )
    PV_output = PV_output.reshape(-1,24)
    WI_output = wind.reshape(-1,24)
    season_daily = np.array([season[i * 24] for i in range(365)])

    return PV_output, WI_output, season_daily

def IL():
    df = pd.read_excel('./data/data_beijing.xlsx', header=None, usecols=[5])
    ILdata=np.array(df).reshape(-1, 24)
    return ILdata

def load_TF():
    df=pd.read_csv('./data/Traffic_2019.csv')
    TFdata=np.array(df).reshape(-1, 24)
    return TFdata

def load_day_type():
    df = pd.read_csv("./data/2019daytype.csv")
    data = np.array(df, dtype=int).reshape(-1)
    return data



class MultiVarTimeSeriesDataset(Dataset):
    """
    Dataset for multivariatr time series with conditional labels.
    each sample is one day of 4-variate time series(PV, Wind, Load, TF), with conditions:
    season(0-3) and day_type(0=workday, 1=non-workday)
    """
    def __init__(self):
        # load data
        self.PV, self.Wind, self.season = Renewable_energy() # (365, 24) / # (365, )
        self.Load = IL()   # (365, 24)
        self.TF = load_TF()  # (365, 24)
        self.daytype = load_day_type()  # (365, )

        self.stats = {}
        self.stats['mean'] = {
            'PV': self.PV.mean(),
            'Wind': self.Wind.mean(),
            'Load': self.Load.mean(),
            'Traffic': self.TF.mean()
        }
        self.stats['std'] = {
            'PV': self.PV.std(),
            'Wind': self.Wind.std(),
            'Load': self.Load.std(),
            'Traffic': self.TF.std()
        }

        # normalize
        self.PV = (self.PV - self.stats['mean']['PV']) / self.stats['std']['PV']
        self.Wind = (self.Wind - self.stats['mean']['Wind']) / self.stats['std']['Wind']
        self.Load = (self.Load - self.stats['mean']['Load']) / self.stats['std']['Load']
        self.TF = (self.TF - self.stats['mean']['Traffic']) / self.stats['std']['Traffic']

    def __len__(self):
        return self.PV.shape[0]  


    def __getitem__(self, idx):
        X_pv = torch.tensor(self.PV[idx], dtype=torch.float32).unsqueeze(0)
        X_wind = torch.tensor(self.Wind[idx], dtype=torch.float32).unsqueeze(0)
        X_load = torch.tensor(self.Load[idx], dtype=torch.float32).unsqueeze(0)
        X_TF = torch.tensor(self.TF[idx], dtype=torch.float32).unsqueeze(0)
        season = torch.tensor(self.season[idx], dtype=torch.long)
        day_type = torch.tensor(self.daytype[idx], dtype=torch.long)
        return X_pv, X_wind, X_load, X_TF, season, day_type
    

if __name__ == "__main__":

    PV, Wind, season = Renewable_energy()
