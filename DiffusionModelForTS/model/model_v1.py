# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: model_v1.py
# @time: 2025/12/26 15:13:23
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: v1


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

v0版本的噪声预测网络存在很大问题，首先虽然对时间步timestep进行了SinusiidalTimeEmbedding，但后续没有经过MLP非线性映射，t不只是一个偏置，
而是控制噪声尺度的核心变量，v0模型对于t的响应较弱，导致模型对所有t预测几乎相同的noise

同时在使用transformer作为encoder时，没有采用位置向量，模型捕捉不到位置信息，可能是导致模型学校不到分布的原因
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
            torch.arange(half_dim, device=device) *
            (-math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class NoisePredictor(nn.Module):
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

        # ========= 1. 每个变量的特征提取 =========
        # (B, 24) -> (B, 24, d_model)
        self.var_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_vars)
        ])

        # ========= 2. 变量 embedding =========
        self.var_emb = nn.Embedding(num_vars, d_model)

        # ========= 3. 时间位置编码（24） =========
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
        assert C == self.num_vars and L == self.seq_len

        device = x_t.device

        # ========= 1. 每个变量独立特征提取 =========
        var_tokens = []
        for i in range(self.num_vars):
            # (B, 24)
            xi = x_t[:, i, :]
            # (B, d_model)
            hi = self.var_proj[i](xi)
            var_tokens.append(hi)

        # (B, 4, d_model)
        var_tokens = torch.stack(var_tokens, dim=1)

        # ========= 2. 展开为 token 序列 =========
        # (B, 4, 24, d_model)
        var_tokens = var_tokens[:, :, None, :].expand(-1, -1, self.seq_len, -1)

        # ========= 3. 加变量 embedding =========
        var_ids = torch.arange(self.num_vars, device=device)
        var_emb = self.var_emb(var_ids)[None, :, None, :]
        var_tokens = var_tokens + var_emb

        # ========= 4. 加时间位置编码 =========
        pos_ids = torch.arange(self.seq_len, device=device)
        pos_emb = self.pos_emb(pos_ids)[None, None, :, :]
        var_tokens = var_tokens + pos_emb

        # ========= 5. reshape 成 Transformer tokens =========
        # (B, 4*24, d_model)
        tokens = var_tokens.reshape(B, self.num_vars * self.seq_len, self.d_model)

        # ========= 6. 条件 embedding（广播） =========
        t_emb = self.time_emb(t)                      # (B, d_model)
        season_emb = self.season_emb(season)          # (B, d_model)
        day_emb = self.daytype_emb(day_type)          # (B, d_model)

        cond_emb = (t_emb + season_emb + day_emb)[:, None, :]
        tokens = tokens + cond_emb

        # ========= 7. Transformer =========
        tokens = self.transformer(tokens)

        # ========= 8. 输出投影 =========
        # (B, 4*24, 1)
        out = self.output_proj(tokens)

        # (B, 4, 24)
        noises_pred = out.view(B, self.num_vars, self.seq_len)

        return noises_pred


