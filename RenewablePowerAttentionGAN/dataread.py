import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

class RenewableEnergyDataset(Dataset):
    def __init__(self, real_data,c):
        self.real_data=real_data
        self.c=c
    def __len__(self):
        return len(self.real_data)

    def __getitem__(self, idx):
        energy_data_sample = self.real_data[idx]
        c_data_sample=self.c[idx]
        return energy_data_sample,c_data_sample

# class sliding_window(Dataset):
#     def __init__(self,data,window_size, step_size):
#         """
#         :param W: 窗口大小，表示每个子序列包含多少个时间步。
#         :param S: 步长，表示窗口每次移动的步。
#         :windows：窗口索引
#         """
#         self.data = data
#         self.window_size = window_size
#         self.step_size = step_size
#         self.windows = self.create_sliding_windows()
#     def create_sliding_windows(self):
#         """
#         生成滑动窗口的索引
#         """
#
#         num_windows = (len(self.data) - self.window_size) // self.step_size + 1
#         windows=np.lib.stride_tricks.as_strided(
#             self.data,
#             shape=(num_windows,self.window_size),
#             strides=(self.data.strides[0]*self.step_size,self.data.strides[0])
#         )
#         print(windows.shape)
#         return windows
#     def __len__(self):
#         return len(self.windows)
#
#     def __getitem__(self, idx):
#         window_data = self.windows[idx]  # 每个窗口的数据
#         return window_data

def sliding_windows(data,window_size,step_size):
    """
    生成滑动窗口的索引
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows=np.lib.stride_tricks.as_strided(
        data,
        shape=(num_windows,window_size),
        strides=(data.strides[0]*step_size,data.strides[0])
    )
    #print(windows.shape)
    return windows

def read_data():
    df = pd.read_csv('10year_energy_data.csv', usecols=['PV', 'Tem'])
    energy = np.array(df['PV'])
    tem = np.array(df['Tem'])
    # data=RenewableEnergyDataset(energy,tem)
    energy_sliding=sliding_windows(energy,720,600)
    tem_sliding=sliding_windows(tem,720,600)
    #print(tem_sliding)
    return energy_sliding,tem_sliding

if __name__=='__main__':

    data=read_data()
    # dataloader = DataLoader(
    #     dataset=read_data(),
    #     batch_size=len(read_data()),
    #     shuffle=False,
    # )
    # for energy_data, c_data in dataloader:
    #
    #     ener=sliding_window(energy_data,720,600)
    #     con=sliding_window(c_data,720,600)
    #     ener.windows=ener.windows.unsqueeze(-1)
    #     print(ener.windows.shape)


        # seq=SequenceEncoder(32)
        # seq_in=seq(ener,con)
        # print(seq_in.shape)
    # test1=sliding_window(data.c,720,672)
    # for batch_data in test1:
    #     print(batch_data.shape)
    # dataloader = DataLoader(
    #     dataset=read_data(),
    #     batch_size=len(read_data()),
    #     shuffle=False,
    # )
    # for real_data, c in dataloader:
    #     print(real_data)
    # for batch_data in dataloader:
    #     print(type(batch_data))  # 打印返回的数据类型
    #     print(batch_data)  # 打印数据内容


    # # 假设输入序列长度为 8760
    # data = torch.randn(100)  # 示例数据，可以替换为你的实际数据
    # z = torch.randn(10, 10, 1)
    # # 定义目标形状
    # batch_size = 10  # 假设批次大小为 32
    # seq_len = 10  # 假设序列长度为 10
    # cond_dim = 1  # 假设条件维度为 27
    #
    # # 检查形状是否匹配
    #
    #
    # # 调整形状为 [batch_size, seq_len, cond_dim]
    # data_reshaped = data.view(batch_size, seq_len, cond_dim)
    #
    # # 创建 DataLoader
    # dataset = TensorDataset(data_reshaped)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #
    # # 测试 DataLoader
    # for batch in dataloader:
    #     print(batch[0].shape)  # 输出形状为 [batch_size, seq_len, cond_dim]
    # print(z)
