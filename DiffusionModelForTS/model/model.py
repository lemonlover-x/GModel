# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: model.py
# @time: 2025/12/25 16:08:02
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: backward process, the noise predict network

"""
模型原理：【时间序列+条件+timestep的条件transformer】，输出噪声ε̂

核心思想：将时间维度L当作token序列，把变量维度C当作通道特征，在transformer中做时间维自注意力和变量维线性投影

输入：
x_t      : (B, C=4, L)
t        : (B,)
season   : (B,)
day_type : (B,)

输出：
eps_pred : (B, C=4, L)
"""


import math
import torch
from torch import nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device) * (-math.log(10000) / (half - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class NoisePredictor(nn.Module):
    def __init__(self, num_vars=4, d_model=128, n_heads=4, n_layers=4, seq_len=24, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(num_vars, d_model)

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.season_emb = nn.Embedding(4, d_model)
        self.daytype_emb = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.output_proj = nn.Linear(d_model, num_vars)

    def forward(self, x_t, t, season, day_type):
        """
        x_t: (B, C, L)
        """
        B, C, L = x_t.shape

        # (B, L, C)
        x = x_t.permute(0, 2, 1)

        # input projection
        x = self.input_proj(x)  # (B, L, d_model)

        # time embedding
        t_emb = self.time_emb(t)[:, None, :]
        x = x + t_emb

        # condition embedding
        cond_emb = (
            self.season_emb(season) + self.daytype_emb(day_type)
        )[:, None, :]
        x = x + cond_emb

        # transformer
        x = self.transformer(x)

        # output projection
        x = self.output_proj(x)  # (B, L, C)

        # back to (B, C, L)
        eps_pred = x.permute(0, 2, 1)

        return eps_pred

