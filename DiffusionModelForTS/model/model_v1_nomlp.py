# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: model_v1.py
# @time: 2025/12/26 15:13:23
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: v1_nomlp


"""
模型原理：【时间序列+条件+timestep的条件transformer】，输出噪声ε̂

输入：
x_t      : (B, C=4, L)
t        : (B,)
season   : (B,)
day_type : (B,)

输出：
eps_pred : (B, C=4, L)

相较于v1模型，该模型是其对照实验的网络设计，即不对变量做mlp特征提取
"""


import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        return: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device)
            * (-math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class NoisePredictor(nn.Module):
    """
    对照实验版本：
    - 不对变量使用 MLP
    - 每个 (variable, time) 的原始标量直接作为 token 输入
    """

    def __init__(
        self,
        num_vars=4,
        seq_len=24,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model

        # ========= 1. 原始数值投影 =========
        # (B, 1) -> (B, d_model)
        self.value_proj = nn.Linear(1, d_model)

        # ========= 2. 变量 embedding =========
        self.var_emb = nn.Embedding(num_vars, d_model)

        # ========= 3. 时间位置编码 =========
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # ========= 4. 条件 embedding =========
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.season_emb = nn.Embedding(4, d_model)
        self.daytype_emb = nn.Embedding(2, d_model)

        # ========= 5. Transformer Encoder =========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # ========= 6. 输出投影 =========
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x_t, t, season, day_type):
        """
        x_t      : (B, 4, 24)
        t        : (B,)
        season   : (B,)
        day_type : (B,)

        return:
        eps_pred : (B, 4, 24)
        """
        B, C, L = x_t.shape
        device = x_t.device
        assert C == self.num_vars and L == self.seq_len

        # ========= 1. 构造 token =========
        # (B, 4, 24, 1)
        tokens = x_t.unsqueeze(-1)

        # (B, 4, 24, d_model)
        tokens = self.value_proj(tokens)

        # ========= 2. 加变量 embedding =========
        var_ids = torch.arange(self.num_vars, device=device)
        var_emb = self.var_emb(var_ids)[None, :, None, :]
        tokens = tokens + var_emb

        # ========= 3. 加时间位置编码 =========
        pos_ids = torch.arange(self.seq_len, device=device)
        pos_emb = self.pos_emb(pos_ids)[None, None, :, :]
        tokens = tokens + pos_emb

        # ========= 4. reshape 成 Transformer token =========
        # (B, 4*24, d_model)
        tokens = tokens.view(B, self.num_vars * self.seq_len, self.d_model)

        # ========= 5. 条件 embedding =========
        t_emb = self.time_emb(t)
        season_emb = self.season_emb(season)
        day_emb = self.daytype_emb(day_type)

        cond_emb = (t_emb + season_emb + day_emb)[:, None, :]
        tokens = tokens + cond_emb

        # ========= 6. Transformer =========
        tokens = self.transformer(tokens)

        # ========= 7. 输出 =========
        out = self.output_proj(tokens)
        noises_pred = out.view(B, self.num_vars, self.seq_len)

        return noises_pred


