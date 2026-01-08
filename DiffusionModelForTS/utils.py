"""
the utils for diffusion model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-21
"""

import os
import torch
import numpy as np
from datetime import datetime, timedelta

def bulid_log_dir(dir):
    curtime = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join(dir, curtime.strftime(f"[%m-%d]%H.%M.%S"))
    # 若文件夹不存在，则创建
    if not os.path.exists(log_path_dir):
        os.makedirs(log_path_dir)
    return log_path_dir


def to_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    else:
        return obj
