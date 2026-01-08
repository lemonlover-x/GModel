# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: evaluate.py
# @time: 2026/01/05 10:50:48
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: evaluation of the generated data


import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist


class Evaluator:
    """
    Evaluator for multivariate conditional time-series generation.

    Input format:
        x_real: (Br, C, L)
        x_fake: (Bf, C, L)
    """

    def __init__(self, var_names=None):
        self.var_names = var_names or ["PV", "Wind", "Load", "Traffic"]

    # ============================================================
    # Basic statistics
    # ============================================================
    def basic_stats(self, x_real, x_fake):
        """
        Mean / Std for each variable
        """
        results = {}

        for i, name in enumerate(self.var_names):
            real = x_real[:, i, :].reshape(-1)
            fake = x_fake[:, i, :].reshape(-1)

            results[name] = {
                "real_mean": real.mean(),
                "fake_mean": fake.mean(),
                "real_std": real.std(),
                "fake_std": fake.std(),
            }

        return results

    # ============================================================
    # Wasserstein distance
    # ============================================================
    def wasserstein(self, x_real, x_fake):
        """
        1D Wasserstein distance for each variable
        """
        results = {}

        for i, name in enumerate(self.var_names):
            real = x_real[:, i, :].reshape(-1)
            fake = x_fake[:, i, :].reshape(-1)

            results[name] = wasserstein_distance(real, fake)

        return results

    # ============================================================
    # DTW (average over random pairs):单条曲线的“形态相似性”,一条生成曲线“像不像”一条真实曲线
    # 这里做的是平均DTW
    # ============================================================
    def dtw_distance(self, x_real, x_fake, num_pairs=100):
        """
        Average DTW distance between real and fake samples
        """
        from scipy.signal import correlate

        def simple_dtw(a, b):
            D = cdist(a[:, None], b[:, None], metric="euclidean")
            dp = np.zeros_like(D)
            dp[0, 0] = D[0, 0]

            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if i == 0 and j == 0:
                        continue
                    dp[i, j] = D[i, j] + min(
                        dp[i - 1, j] if i > 0 else np.inf,
                        dp[i, j - 1] if j > 0 else np.inf,
                        dp[i - 1, j - 1] if (i > 0 and j > 0) else np.inf,
                    )
            return dp[-1, -1]

        results = {}
        Br = x_real.shape[0]
        Bf = x_fake.shape[0]

        for c, name in enumerate(self.var_names):
            dists = []
            for _ in range(num_pairs):
                i = np.random.randint(Br)
                j = np.random.randint(Bf)

                d = simple_dtw(
                    x_real[i, c].cpu().numpy(),
                    x_fake[j, c].cpu().numpy(),
                )
                dists.append(d)

            results[name] = float(np.mean(dists))

        return results

    # ============================================================
    # Correlation structure distance
    # 多变量之间的结构一致性，越小越好
    # ============================================================
    def correlation_distance(self, x_real, x_fake):
        """
        Frobenius norm between average correlation matrices
        """
        def avg_corr(x):
            # x: (B, C, L)
            corrs = []
            for i in range(x.shape[0]):
                corr = np.corrcoef(x[i].cpu().numpy())
                corrs.append(corr)
            return np.mean(corrs, axis=0)

        corr_real = avg_corr(x_real)
        corr_fake = avg_corr(x_fake)

        return np.linalg.norm(corr_real - corr_fake, ord="fro")

    # ============================================================
    # Conditional MMD (RBF kernel)
    # 联合分布层面的差异
    # ============================================================
    def conditional_mmd(self, x_real, x_fake, sigma=1.0):
        """
        MMD between real and fake under same condition
        """
        def rbf(x, y):
            dists = cdist(x, y, metric="sqeuclidean")
            return np.exp(-dists / (2 * sigma ** 2))

        real = x_real.reshape(x_real.shape[0], -1).cpu().numpy()
        fake = x_fake.reshape(x_fake.shape[0], -1).cpu().numpy()

        Krr = rbf(real, real).mean()
        Kff = rbf(fake, fake).mean()
        Krf = rbf(real, fake).mean()

        return Krr + Kff - 2 * Krf
    
    # ============================================================
    # Energy Score (variable-wise)
    # ============================================================
    def energy_score(self, x_real, x_fake):
        """
        Energy Score computed separately for each variable.

        Args:
            x_real: torch.Tensor, (Br, C, L)
            x_fake: torch.Tensor, (Bf, C, L)

        Returns:
            results: dict {var_name: float}
        """
        x_real = x_real.cpu().numpy()
        x_fake = x_fake.cpu().numpy()

        Br, C, L = x_real.shape
        Bf = x_fake.shape[0]

        results = {}

        for c, name in enumerate(self.var_names):
            real = x_real[:, c, :]   # (Br, L)
            fake = x_fake[:, c, :]   # (Bf, L)

            # ===== term 1: E||X - Y|| =====
            # shape: (Bf, Br)
            dist_rf = cdist(fake, real, metric="euclidean")
            term_real = dist_rf.mean()

            # ===== term 2: E||X - X'|| =====
            # shape: (Bf, Bf)
            dist_ff = cdist(fake, fake, metric="euclidean")

            # remove diagonal (i == j)
            mask = ~np.eye(Bf, dtype=bool)
            term_fake = dist_ff[mask].mean()

            # ===== energy score =====
            es = term_real - 0.5 * term_fake
            results[name] = float(es)

        return results

    # ============================================================
    # Coverage Rate (CR)
    # 覆盖真实样本的比例，越大越好
    # ============================================================
    def coverage_rate(self, x_real, x_fake, alpha=0.9):
        """
        Coverage Rate (CR) for each variable.

        Args:
            x_real: torch.Tensor, (Br, C, L)
            x_fake: torch.Tensor, (Bf, C, L)
            alpha: confidence level

        Returns:
            results: dict {var_name: float}
        """
        x_real = x_real.cpu().numpy()
        x_fake = x_fake.cpu().numpy()

        Br, C, L = x_real.shape
        total = Br * L

        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q

        results = {}

        for c, name in enumerate(self.var_names):
            real = x_real[:, c, :]   # (Br, L)
            fake = x_fake[:, c, :]   # (Bf, L)

            covered = 0

            for t in range(L):
                q_low = np.quantile(fake[:, t], lower_q)
                q_high = np.quantile(fake[:, t], upper_q)

                covered += np.sum(
                    (real[:, t] >= q_low) & (real[:, t] <= q_high)
                )

            results[name] = float(covered / total)

        return results
    
    # ============================================================
    # Average Width of Power Interval (AWPI)
    # 不确定性刻画能力，越小越好（在 CR 相近时）
    # ============================================================
    def awpi(self, x_fake, alpha=0.9):
        """
        Average Width of Power Interval (AWPI) for each variable.

        Args:
            x_fake: torch.Tensor, (Bf, C, L)
            alpha: confidence level

        Returns:
            results: dict {var_name: float}
        """
        x_fake = x_fake.cpu().numpy()

        Bf, C, L = x_fake.shape

        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q

        results = {}

        for c, name in enumerate(self.var_names):
            fake = x_fake[:, c, :]
            width_sum = 0.0

            for t in range(L):
                q_low = np.quantile(fake[:, t], lower_q)
                q_high = np.quantile(fake[:, t], upper_q)
                width_sum += (q_high - q_low)

            results[name] = float(width_sum / L)

        return results

    # ============================================================
    # Unified evaluation
    # ============================================================
    def evaluate(self, x_real, x_fake):
        """
        Run all metrics
        """
        results = {
            "basic_stats": self.basic_stats(x_real, x_fake),
            "wasserstein": self.wasserstein(x_real, x_fake),
            "energy_score": self.energy_score(x_real, x_fake),
            "dtw": self.dtw_distance(x_real, x_fake),
            "correlation_distance": self.correlation_distance(x_real, x_fake),
            "conditional_mmd": self.conditional_mmd(x_real, x_fake),
            "coverage_rate": self.coverage_rate(x_real, x_fake, alpha=0.9),
            "awpi": self.awpi(x_fake, alpha=0.9),
        }
        return results
        
    

