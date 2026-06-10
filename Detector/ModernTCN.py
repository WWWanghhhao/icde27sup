import os
import argparse
import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Detector.ModernTCN_layers.ModernTCN_base import ModernTCN


def save_npy(data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset/GECCO/")
    parser.add_argument("--output_dir", type=str, default="./moderntcn_outputs/GECCO")

    # DualTF 超参数
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--enc_in", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--dims", type=int, default=8)
    parser.add_argument("--anomaly_ratio", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [f'col_{i}' for i in range(args.enc_in)]

    train_medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)

    if 'label' in train_df.columns:
        train_df = train_df.drop(columns=['label'])
    if 'label' in test_df.columns:
        test_df = test_df.drop(columns=['label'])
    if 'timestamp' in train_df.columns:
        train_df = train_df.drop(columns=['timestamp'])
    if 'timestamp' in test_df.columns:
        test_df = test_df.drop(columns=['timestamp'])

    # 训练/验证切分（不打乱时间顺序）

    # 训练
    model = ModernTCN(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        enc_in=args.enc_in,
        anomaly_ratio=args.anomaly_ratio
    )
    print('start training...')
    model.detect_fit(train_df, test_df)

    # 按原实现计算异常分数（1D）
    print('start inferencing...')
    train_score = model.detect_score(train_df)
    # print(train_df.shape)
    # print(train_score.shape)
    test_score = model.detect_score(test_df)
    # print(test_df.shape)
    # print(test_score.shape)

    # 保存
    save_npy(
        train_score,
        os.path.join(args.output_dir, "train_score.npy"),
    )
    save_npy(
        test_score,
        os.path.join(args.output_dir, "test_score.npy"),
    )

    print(f"Saved train_score.npy and test_score.npy to {args.output_dir}")


if __name__ == "__main__":
    main()
