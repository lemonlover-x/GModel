# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: forward.py
# @time: 2025/12/25 14:50:06
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: 前向加噪过程，独立加噪

import torch
from torch import nn
import torch.nn.functional as F

class ForwardProcess(nn.Module):
    """
    Forward diffusion process for multivariate time series.
    x_0 shape: (B, C, L)
    """
    def __init__(self, T, start=0.0001, end=0.02):
        super().__init__()

        self.T = T

        # beta schedule
        betas = torch.linspace(start, end, T)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # ===== register buffers (关键) =====
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt(1.0 / alphas)
        )

        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1. - alphas_cumprod)
        )

        self.register_buffer(
            "posterior_variance",
            betas * (1. - alphas_cumprod_prev)
            / (1. - alphas_cumprod)
        )

    def get_index_from_list(self, vals, t, x_shape):
        """
        vals: (T,)
        t: (B,)
        return: (B, 1, 1)
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x_0, t):
        """
        x_0: (B, C, L)
        t:   (B,)
        """
        noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise
