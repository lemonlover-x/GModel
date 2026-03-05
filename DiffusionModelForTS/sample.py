# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: generate.py
# @time: 2025/12/25 17:10:36
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: Sample the data based on a trained model

import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.model_v1_nomlp import NoisePredictor

from evaluate import Evaluator
from utils import to_json_serializable

from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset


class Sampler:
    """"""
    def __init__(self, arg_dict, model, fp) -> None:
        
        self.arg_dict = arg_dict
        self.model = model
        self.fp = fp

        # betas parameters of forward process
        self.betas = fp.betas
        self.sqrt_one_minus_alphas_cumprod = fp.sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = fp.sqrt_recip_alphas
        self.sqrt_alphas_cumprod = fp.sqrt_alphas_cumprod
        # p[x(t-1)|x(t)]下的方差
        self.posterior_variance = fp.posterior_variance

        # stepsize
        self.stepsize = 30


    @torch.no_grad()
    def sample(self):
        B = self.arg_dict['num']
        C = 4
        L = 24
        T = self.arg_dict['T']

        # 初始化噪声
        xt = torch.randn(size=(B, C, L))

        # 条件输入
        season = torch.full((B,), self.arg_dict['season'], dtype=torch.long)
        day_type = torch.full((B,), self.arg_dict['day_type'], dtype=torch.long)

        for t in tqdm(reversed(range(T)), desc='Sampling'):
            t_batch = torch.full((B,), t, dtype=torch.long)

            # 预测当前噪声
            noise_pred = self.model(xt, t_batch, season, day_type)

            # 估计 x0
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

            if t > 0:
                # 后验均值 μ_t
                alpha_t = self.fp.alphas[t]
                alpha_cumprod_prev = self.fp.alphas_cumprod[t-1]
                beta_t = self.fp.betas[t]

                mu_t = (torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - self.fp.alphas_cumprod[t])) * x0_pred \
                    + (torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - self.fp.alphas_cumprod[t])) * xt

                # 后验方差 σ_t
                sigma_t = torch.sqrt(self.fp.posterior_variance[t])

                # 采样 x_{t-1}
                xt = mu_t + sigma_t * torch.randn_like(xt)
            else:
                xt = x0_pred

        return xt



def get_timeseries_by_condition(
    dataset,
    season=None,
    day_type=None,
    denorm=True,
):
    """
    从 MultiVarTimeSeriesDataset 中：
    - 按条件筛选
    - 拼接 4 个变量
    - 可选反归一化

    return:
        X: (N, 4, 24)
        season: (N,)
        day_type: (N,)
    """
    X_list, season_list, day_list = [], [], []

    for i in range(len(dataset)):
        s = dataset.season[i]
        d = dataset.daytype[i]

        if season is not None and s != season:
            continue
        if day_type is not None and d != day_type:
            continue

        X_pv, X_wind, X_load, X_TF, _, _ = dataset[i]
        X = torch.cat([X_pv, X_wind, X_load, X_TF], dim=0)  # (4, 24)

        X_list.append(X)
        season_list.append(torch.tensor(s))
        day_list.append(torch.tensor(d))

    if len(X_list) == 0:
        raise ValueError("No samples match the given condition.")

    X = torch.stack(X_list)
    season = torch.stack(season_list)
    day_type = torch.stack(day_list)

    # ========= 反归一化 =========
    if denorm:
        stats = dataset.stats
        X[:, 0] = X[:, 0] * stats['std']['PV'] + stats['mean']['PV']
        X[:, 1] = X[:, 1] * stats['std']['Wind'] + stats['mean']['Wind']
        X[:, 2] = X[:, 2] * stats['std']['Load'] + stats['mean']['Load']
        X[:, 3] = X[:, 3] * stats['std']['Traffic'] + stats['mean']['Traffic']

    return X

def denormalize_timeseries(x, stats):
    """
    x: torch.Tensor, (B, 4, L), normalized
    stats: dataset.stats
    """
    x = x.clone()

    x[:, 0] = x[:, 0] * stats['std']['PV'] + stats['mean']['PV']
    x[:, 1] = x[:, 1] * stats['std']['Wind'] + stats['mean']['Wind']
    x[:, 2] = x[:, 2] * stats['std']['Load'] + stats['mean']['Load']
    x[:, 3] = x[:, 3] * stats['std']['Traffic'] + stats['mean']['Traffic']

    return x

def save_generated_by_condition(
    x_fake,
    season,
    day_type,
    save_root="./results/generated",
    save_format="csv",  # "npy" or "csv"
):
    """
    x_fake: torch.Tensor, (B, 4, 24), 已反归一化
    season: int
    day_type: int 
    """

    assert x_fake.ndim == 3 and x_fake.shape[1] == 4

    var_names = ["PV", "Wind", "Load", "Traffic"]

    # ===== 保存路径 =====
    save_dir = os.path.join(
        save_root,
        f"season_{season}_daytype_{day_type}"
    )
    os.makedirs(save_dir, exist_ok=True)

    x_fake = x_fake.cpu().numpy()  # (B, 4, 24)

    for i, var in enumerate(var_names):
        data = x_fake[:, i, :]  # (B, 24)

        file_path = os.path.join(save_dir, f"{var}.{save_format}")

        if save_format == "npy":
            np.save(file_path, data)

        elif save_format == "csv":
            # 每一行是一条样本（24小时）
            np.savetxt(
                file_path,
                data,
                delimiter=",",
                fmt="%.6f"
            )

        else:
            raise ValueError(f"Unsupported save format: {save_format}")

    print(f"Generated data saved to: {save_dir}")



def plot_generated_timeseries(
    x_fake,
    x_real=None,
    fake_indices=None,
    real_indices=None,
):
    """
    x_fake: torch.Tensor, (Bf, 4, L)
    x_real: torch.Tensor, (Br, 4, L) or None
    fake_indices: list[int]
    real_indices: list[int]
    """

    if fake_indices is None:
        fake_indices = list(range(min(100, x_fake.shape[0])))

    if x_real is not None and real_indices is None:
        real_indices = list(range(min(100, x_real.shape[0])))

    x_fake = x_fake.cpu().numpy()
    if x_real is not None:
        x_real = x_real.cpu().numpy()

    channels = ['PV', 'Wind', 'Load', 'TF']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    for c in range(4):

        # ===== fake 数据 =====
        for idx in fake_indices:
            axes[c].plot(
                x_fake[idx, c],
                color='tab:blue',
                alpha=0.6,
                linewidth=2,
                label='Fake' if idx == fake_indices[0] else None
            )

        # ===== real 数据 =====
        if x_real is not None:
            for idx in real_indices:
                axes[c].plot(
                    x_real[idx, c],
                    color='tab:orange',
                    alpha=0.6,
                    linewidth=2,
                    linestyle='--',
                    label='Real' if idx == real_indices[0] else None
                )

        axes[c].set_title(channels[c])
        axes[c].set_ylabel(channels[c])
        axes[c].grid(True)
        axes[c].legend()

    axes[2].set_xlabel('Time step')
    axes[3].set_xlabel('Time step')

    plt.tight_layout()
    plt.savefig('./results/fake.svg', format="svg", bbox_inches='tight', dpi=300)

def data_filter(
    data,
    season=None,
    day_type=None,
    low_ratio=0.2,
    high_quantile=0.9,
    low_quantile=0.1,
    day_start=6,
    day_end=18,
):
    """
    基于真实光伏日分布特征的过滤：
    1. 白天段不允许明显负值（允许小噪声）
    2. 形态异常过滤（剧烈振荡）
    3. 季节统计约束（你原逻辑保留）
    """

    pv = data[:, 0, :]  # (B, 24)

    # ========= 1. 白天物理合理性约束 =========
    pv_day = pv[:, day_start:day_end]
    pv_max = pv.max(dim=1).values

    # 允许 2% 峰值的负噪声
    neg_tol = 0.02 * pv_max
    valid_mask = pv_day.min(dim=1).values >= -neg_tol
    data = data[valid_mask]

    pv = data[:, 0, :]
    pv_max = pv.max(dim=1).values

    # ========= 2. 形态异常过滤 =========
    tv = torch.mean(torch.abs(pv[:, 1:] - pv[:, :-1]), dim=1)
    tv_thr = torch.quantile(tv, 0.98)
    data = data[tv <= tv_thr]

    pv = data[:, 0, :]
    pv_max = pv.max(dim=1).values

    # ========= 3. 季节统计过滤 =========
    if season == 1:  # summer
        low_thr = torch.quantile(pv_max, low_quantile)

        high_mask = pv_max >= low_thr
        low_mask = pv_max < low_thr

        x_high = data[high_mask]
        x_low = data[low_mask]

        if x_low.shape[0] > 0:
            keep_num = max(1, int(low_ratio * x_low.shape[0]))
            idx = torch.randperm(x_low.shape[0])[:keep_num]
            x_low = x_low[idx]

        x_filtered = torch.cat([x_high, x_low], dim=0)

    elif season == 3:  # winter
        high_thr = torch.quantile(pv_max, high_quantile)
        x_filtered = data[pv_max <= high_thr]

    else:
        x_filtered = data

    return x_filtered

def main(arg_dict):
    """"""
    checkpoints = torch.load(arg_dict['checkpoints'])
    model = NoisePredictor()
    model.load_state_dict(checkpoints)

    # real data
    dataset = MultiVarTimeSeriesDataset()
    real = get_timeseries_by_condition(dataset, season=arg_dict["season"], day_type=arg_dict["day_type"])
    # print(len(real))

    # forward process
    fp = ForwardProcess(arg_dict['T'])

    # sampler
    sampler = Sampler(arg_dict, model, fp)

    x_fake = sampler.sample()
    # ===== 反归一化 =====
    x_fake1 = denormalize_timeseries(x_fake, dataset.stats)

    # 数据过滤
    x_fake1 = data_filter(x_fake1)
    print(len(x_fake1))

    # 评估
    evaluator = Evaluator()
    rea_norm = get_timeseries_by_condition(dataset, season=arg_dict["season"], day_type=arg_dict["day_type"], denorm=False)
    results = evaluator.evaluate(rea_norm, x_fake)
    results_json = to_json_serializable(results)
    with open(f"evaluation_results/evaluation_season_{arg_dict['season']}_daytype_{arg_dict['day_type']}.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    # 保存数据
    save_generated_by_condition(
        x_fake1,
        season=arg_dict["season"],
        day_type=arg_dict["day_type"],
        save_root="./results/generated",
        save_format="csv",  # 或 "npy"
    )
    
    # plot
    plot_generated_timeseries(x_fake1, real)



if __name__ == '__main__': 

    arg_dict = {
        "checkpoints": './logs/model_v1_nomlp_epoch_5000/model_4999.tar', 
        "num": 500,  # the number of data you want to generate
        "T": 200,
        "season": 3,
        "day_type": 1,
    }

    main(arg_dict)
