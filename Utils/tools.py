import os

import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import logging, sys


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        args.logger.info('Updating learning rate to {}'.format(lr))




class SlidingWindowDataset(Dataset):
    def __init__(self, data_array: np.ndarray, seq_len: int):
        self.data = torch.from_numpy(data_array.astype(np.float32))
        self.seq_len = seq_len
        self.n_samples = max(0, len(data_array) - seq_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 实时切片，内存占用极低
        window = self.data[idx: idx + self.seq_len]
        # window = window + torch.rand_like(window) * 1e-5  # 避免全0输入
        return window, 0  # 0是dummy target


def provide_data_loader(data, seq_len: int, batch_size: int, shuffle: bool = False, num_workers=0) -> DataLoader:
    dataset = SlidingWindowDataset(data_array=data, seq_len=seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)
    return data_loader


class ForecastingDataset(Dataset):
    """
    时序预测任务的数据集
    X: [t : t + seq_len]
    Y: [t + seq_len - label_len : t + seq_len + pred_len]
    """

    def __init__(self, data_array: np.ndarray, seq_len: int, label_len: int, pred_len: int):
        """
        Args:
            data_array: 原始时序数据
            seq_len: 输入序列长度
            label_len: 标签中与输入重叠的部分长度
            pred_len: 需要预测的未来时间步长度
        """
        self.data = data_array.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # 计算有效样本数
        # 需要保证: t + seq_len + pred_len <= len(data)
        self.n_samples = max(0, len(data_array) - seq_len - pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_x = torch.from_numpy(self.data[idx: idx + self.seq_len].copy())
        seq_y = torch.from_numpy(
            self.data[idx + self.seq_len - self.label_len:
                      idx + self.seq_len + self.pred_len].copy()
        )
        return seq_x, seq_y


def provide_forecasting_data_loader(
        data: np.ndarray,
        seq_len: int,
        label_len: int,
        pred_len: int,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0
) -> DataLoader:
    """
    为时序预测任务提供 DataLoader

    Args:
        data: 输入的时序数据数组
        seq_len: 输入序列长度
        label_len: 标签中与输入重叠的部分长度
        pred_len: 需要预测的未来时间步长度
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数

    Returns:
        DataLoader 对象
    """
    dataset = ForecastingDataset(
        data_array=data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader





def get_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlidingWindowDataset2(Dataset):
    def __init__(self, data_array: np.ndarray, mp_array: np.ndarray, seq_len: int):
        """
        滑动窗口数据集，同时返回数据和对应的mp标签

        Args:
            data_array: 时间序列数据，形状为 (n_timesteps, n_features)
            mp_array: 元模式数据，形状为 (n_timesteps, mp_features) 或 (n_timesteps,)
            seq_len: 序列长度
        """
        assert len(data_array) == len(mp_array), "Data and MP arrays must have same length"

        self.data = torch.from_numpy(data_array.astype(np.float32))
        self.mp = torch.from_numpy(mp_array.astype(np.float32))
        self.seq_len = seq_len
        self.n_samples = max(0, len(data_array) - seq_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 获取数据窗口
        data_window = self.data[idx: idx + self.seq_len]

        # 获取对应的mp窗口（如果mp是二维的，保持相同的窗口）
        if len(self.mp.shape) == 1:
            mp_window = self.mp[idx: idx + self.seq_len]
        else:
            mp_window = self.mp[idx: idx + self.seq_len, :]

        return data_window, mp_window


def provide_data2_loader(data, mp_data, seq_len: int, batch_size: int,
                         shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    创建数据加载器，同时加载数据和mp标签
    """
    dataset = SlidingWindowDataset2(data_array=data, mp_array=mp_data, seq_len=seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             drop_last=True, num_workers=num_workers)
    return data_loader


def reconstruct_from_windows(all_windows: np.ndarray, original_length: int, seq_len: int):
    """
    从滑动窗口重构原始序列，对重叠部分取平均

    Args:
        all_windows: shape [n_windows, seq_len, n_features] 或 [n_windows, seq_len]
        original_length: 原始序列长度
        seq_len: 窗口长度

    Returns:
        reconstructed: shape [original_length, n_features] 或 [original_length]
    """
    n_windows = all_windows.shape[0]

    # 处理多维和一维情况
    # 使用半精度进行累积计算
    if all_windows.ndim == 3:
        n_features = all_windows.shape[2]
        reconstructed = np.zeros((original_length, n_features), dtype=np.float16)
        counts = np.zeros((original_length, n_features), dtype=np.float16)
    else:
        reconstructed = np.zeros(original_length, dtype=np.float16)
        counts = np.zeros(original_length, dtype=np.float16)

    # 累加所有窗口
    for i in range(n_windows):
        start_idx = i
        end_idx = i + seq_len
        reconstructed[start_idx:end_idx] += all_windows[i]
        counts[start_idx:end_idx] += 1

    # 对重叠部分取平均
    reconstructed = reconstructed / np.maximum(counts, 1)  # 避免除0
    reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=1e6, neginf=-1e6)
    reconstructed = np.clip(reconstructed, -1e6, 1e6)
    return reconstructed

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)