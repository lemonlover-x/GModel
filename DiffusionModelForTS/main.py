# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: main.py
# @time: 2025/12/25 14:57:43
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: the train script


import os
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model_v1 import NoisePredictor
from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset
from utils import bulid_log_dir




class Trainer:
    """"""
    def __init__(self, hyper_params, fp, model, optimizer, dataloader, device):
        """"""
        self.hyper_params = hyper_params
        self.train_loader = dataloader
        self.device = device
        self.fp = fp
        self.model = model
        self.optimizer = optimizer
        # tensorboard writer
        self.writer = SummaryWriter(self.hyper_params['log_dir'])
        # loss func
        self.loss_func = F.mse_loss

    def _train_for_epoch(self, epoch):
        """"""
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
        loop.set_description(f'epoch {epoch}')
        loss_list = []

        for (X_pv, X_wind, X_load, X_TF, season, day_type) in loop:
            X_pv = X_pv.to(self.device)
            X_wind = X_wind.to(self.device)
            X_load = X_load.to(self.device)
            X_TF = X_TF.to(self.device)
            season = season.to(self.device)
            day_type = day_type.to(self.device)

            # forward process
            # generate the random t
            """
            我们在训练时，所用的t是随机采样的，这样做的目的是：让模型在所有噪声强度上都学会去噪
            为什么t不是一步一步增加？
            从概率模型的角度，训练不是在模拟轨迹，而是在学习条件分布
            从优化目标的角度来看，随机采样的t才是无偏估计
            """
            # concat
            X = torch.concat([X_pv, X_wind, X_load, X_TF], dim=1)
            t = torch.randint(0, self.hyper_params["T"], size=(X.shape[0], )).to(self.device)
            noisy_X, noises = self.fp(X, t)

            # backward process
            noise_pred = self.model(noisy_X, t, season, day_type)

            loss = self.loss_func(noises, noise_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            loss_list.append(round(float(loss.detach().cpu().numpy()), 3))
        
        return sum(loss_list) / len(loss_list)


    def train(self):
        """"""
        for epoch in range(self.hyper_params['num_epochs']):

            # train for epoch
            loss = self._train_for_epoch(epoch)


            if (epoch+1) % 500 == 0:
                self._save_model(epoch, min=True)


            # 记录日志
            self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=epoch)

    def _save_model(self, epoch, min=False):
        """
        """
        if not os.path.exists(self.hyper_params['log_dir']):
            os.makedirs(self.hyper_params['log_dir'])
        checkpoints = self.model.state_dict()
        path = self.hyper_params['log_dir'] + f'/model_{epoch}.tar'
        print(f'==> Saving checkpoints: {epoch}')
        torch.save(checkpoints, path)



# hyper parameter
hyper_params = {
    "T": 200,   # time steps
    "batch_size": 16,
    "lr": 2e-5,
    "num_epochs": 5000,
    "log_dir": bulid_log_dir(dir='./logs')
}


# create the dataset
dataset = MultiVarTimeSeriesDataset()
dataloader = DataLoader(
)
# gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# forward process model
fp = ForwardProcess(hyper_params["T"]).to(device)

# noise predictor
model = NoisePredictor().to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'])

trainer = Trainer(hyper_params, fp, model, optimizer, dataloader, device)
trainer.train()