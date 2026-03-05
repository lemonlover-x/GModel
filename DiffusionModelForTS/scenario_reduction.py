# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: scenario_reduction.py
# @time: 2026/02/07 16:01:10
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: scenatio reduction based on pv


import os
import json
import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance


def evaluate_clusters(pv_scenarios, basedir):
    """"""
    K = [2, 3, 4, 5, 6, 7, 8, 9]

    wasserstein_inertias = []

    for k in K:
        kmedoids = KMedoids(
            n_clusters=k, 
            metric="euclidean",
            method="pam",
            random_state=0
        ).fit(pv_scenarios)

        labels = kmedoids.labels_
        medoid_indices = kmedoids.medoid_indices_
        medoids = pv_scenarios[medoid_indices]

        # 计算簇内wasserstein距离
        w_dis = 0.0
        for i in range(len(pv_scenarios)):
            c = labels[i]
            w_dis += wasserstein_distance(
                pv_scenarios[i],
                medoids[c]
            )

        wasserstein_inertias.append(w_dis)

    plt.figure(figsize=(6, 4))
    plt.plot(K, wasserstein_inertias, marker="o")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'evaluate_clusters.svg'), format="svg", bbox_inches='tight', dpi=300)


if __name__ == "__main__":

    data_columns = ["PV", "Load", "Traffic", "Wind"]

    base_dir = "results/generated/season_3_daytype_1"
    pv_scenarios = pd.read_csv(os.path.join(base_dir, f'{data_columns[0]}.csv')).values
    load_scenarios = pd.read_csv(os.path.join(base_dir, f'{data_columns[1]}.csv')).values
    tf_scenarios = pd.read_csv(os.path.join(base_dir, f'{data_columns[2]}.csv')).values
    wind_scenarios = pd.read_csv(os.path.join(base_dir, f'{data_columns[3]}.csv')).values


    # 对数据进行归一化
    X_min = pv_scenarios.min()
    X_max = pv_scenarios.max()
    pv_normed = (pv_scenarios - X_min) / (X_max - X_min)

    # 选择最佳簇的数量
    evaluate_clusters(pv_normed, base_dir)


    # ********************   根据最佳簇的数量保存最后结果   *********************** #

    best_k = 8

    kmedoids = KMedoids(
        n_clusters=best_k, 
        metric="euclidean",
        method="pam",
        random_state=0
    ).fit(pv_scenarios)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_

    # 获取每一类的概率
    cluster_counts = np.bincount(labels, minlength=best_k)
    probs = [round(item/len(labels), 4) for item in cluster_counts]

    # 保存中心点数据
    pv_medoids = pv_scenarios[medoid_indices]
    load_medoids = load_scenarios[medoid_indices]
    tf_medoids = tf_scenarios[medoid_indices]
    wind_medoids = wind_scenarios[medoid_indices]

    # pv数据后处理: 小于0.02的都置为0
    pv_medoids[pv_medoids < 0.02] = 0

    result = {
        "n_clusters": best_k,
        "clusters": []
    }

    for i in range(best_k):
        result["clusters"].append({
            "cluster_id": int(i),
            "probability": float(probs[i]),
            "medoid_index": int(medoid_indices[i]),
            "pv": pv_medoids[i].tolist(),
            "load": load_medoids[i].tolist(),
            "traffic": tf_medoids[i].tolist(),
            "wind": wind_medoids[i].tolist()
        })

    with open(os.path.join(base_dir, "cluster_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False, separators=(',', ':'))
        








    




