import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math


class SE(nn.Module):
    def __init__(self, in_channels, ratio):
        super().__init__()
        # 确保 reduction 后至少有 1 个 channel
        reduced_channels = max(1, in_channels // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.se(X)


class SENet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, ratio):
        super().__init__()
        # Padding 设为 'same' 模式或适当值以防止尺寸缩减过快，这里简单使用默认0
        # 注意：对于时间序列 Conv2d，通常 kernel_size=(height, width) 对应 (time, features)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.se = SE(out_channels, ratio)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = self.conv(X)
        coef = self.se(X)
        X = X * coef
        return self.tanh(X)


class SELSTMModel_v3(nn.Module):
    def __init__(self, num_features, seq_len, hidden_size=64, num_layers=3,
                 scales=(64, 128, 192), fuse_mode="max"):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.scales = scales
        self.fuse_mode = fuse_mode

        mid = max(1, num_features // 4)

        self.feature_se = nn.Sequential(
            nn.Linear(num_features, mid),
            nn.ReLU(),
            nn.Linear(mid, num_features),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

        self.reconstructor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size // 2),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size // 2, out_features=num_features),
        )

        # ---------- 参数初始化 ----------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, X):
        """
        输入 X: (Batch, Seq_Len, Num_Features)
        计算流程：经过 SENet 提取 -> LSTM 序列建模 -> 重建
        """
        se_weights = self.feature_se(X)  # (B, L, N)
        X_weighted = X * se_weights  # (B, L, N)
        lstm_out, _ = self.lstm(X_weighted)  # (B, L, H)
        reconstructed = self.reconstructor(lstm_out)  # (B, L, N)
        return reconstructed

    def forward_with_multiscale(self, X):
        """
        多尺度 residual 融合版 forward
        返回:
            fused_residual: (B, L, N)
            last_residual:  (B, L, N)  # 最后一个尺度的 residual
        """
        residuals = []
        for s in self.scales:
            if s > self.seq_len:
                continue
            x = X.clone()
            if s < self.seq_len:
                x[:, s:, :] = 0  # 将后 seq_len - s 步置零
                # keep_start = self.seq_len - s  # 开始保留的起始索引
                # x[:, :keep_start, :] = 0  # 将前 keep_start 步置零
            # 当 s == self.seq_len 时，不做任何 mask，保留全部序列
            rec = self.forward(x)
            res = torch.abs(rec - x)
            residuals.append(res)

        # 融合
        if self.fuse_mode == "max":
            fused = torch.stack(residuals, dim=0).amax(dim=0)
        elif self.fuse_mode == "min":
            fused = torch.stack(residuals, dim=0).amin(dim=0)
        else:
            fused = torch.stack(residuals, dim=0).mean(dim=0)

        return fused, residuals[-1]


class SELSTMModel2(nn.Module):
    def __init__(self, num_features, seq_len, hidden_size=64, single_output=False, num_layers=3):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.single_output = single_output

        # SENet: 输入 (Batch, 1, Seq_Len, Num_Features)
        kernel_width = min(3, num_features)
        self.senet = SENet(in_channels=1,
                           out_channels=16,
                           kernel_size=(1, kernel_width),
                           stride=(1, 1),
                           ratio=4)

        # 通过一次虚拟前向确定 LSTM 输入维度（保持与之前逻辑一致）
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_len, num_features)
            dummy_output = self.senet(dummy_input)
            b, c, h, w = dummy_output.shape
            lstm_input_size = c * w
            # 卷积后时间维度（应等于 seq_len，如果 SENet 不改变高度）
            self.lstm_seq_len = h

        # LSTM 接受每个时间步的扁平特征向量
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers)

        # 重建层：将 LSTM 每个时间步的 hidden 映射回原始特征维度
        self.reconstructor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=num_features),
        )

    def forward(self, X):
        """
        输入 X: (Batch, Seq_Len, Num_Features)
        返回 anomaly scores:
            - 如果 single_output=True: (Batch, Seq_Len, 1)
            - 否则: (Batch, Seq_Len, Num_Features)
        计算流程：经过 SENet 提取 -> LSTM 序列建模 -> 重建 -> 与原始 X 的逐时刻平方误差作为异常分数
        """
        orig = X  # (B, S, F)

        # 增加 channel 维度 -> (B, 1, S, F)
        X = X.unsqueeze(1)

        # SENet -> (B, C, S, W)
        X = self.senet(X)

        # 变换为 LSTM 输入: (B, S, C*W)
        X = X.permute(0, 2, 1, 3)
        b, s, c, w = X.shape
        X = X.reshape(b, s, c * w)

        # LSTM 输出序列: out (B, S, Hidden)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(X)

        # 对每个时间步做重建 -> recon (B, S, Num_Features)
        recon = self.reconstructor(out)

        # 逐时刻、逐特征的平方误差
        sq_err = (recon - orig) ** 2  # (B, S, F)

        if self.single_output:
            # 在特征维度上求均值，返回 (B, S, 1)
            scores = torch.mean(sq_err, dim=2, keepdim=True)
        else:
            # 返回每个特征的误差 (B, S, F)
            scores = sq_err

        return scores


class SELSTMModel_v4(nn.Module):
    def __init__(self, num_features, seq_len, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # 经典 SE: 先在时间维聚合
        mid = max(1, num_features // 4)
        self.feature_se = nn.Sequential(
            nn.Linear(num_features, mid),
            nn.ReLU(),
            nn.Linear(mid, num_features),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 更强一点的重构器 + 残差
        self.reconstructor = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, lstm_out_dim * 2),
            nn.GELU(),
            nn.Linear(lstm_out_dim * 2, num_features),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                # 对 LSTM 的 4 组权重矩阵统一初始化
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:  # 输入→隐藏
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:  # 隐藏→隐藏
                        nn.init.orthogonal_(param.data)  # 正交保持梯度范数
                    elif 'bias' in name:  # 4 个门的偏置
                        nn.init.zeros_(param.data)

    def forward(self, X):
        # X: (B, L, N)
        # 时间维全局池化做 SE
        se_input = X.mean(dim=1)  # (B, N)
        se_weights = self.feature_se(se_input).unsqueeze(1)  # (B, 1, N)
        X_weighted = X * se_weights  # (B, L, N)

        lstm_out, _ = self.lstm(X_weighted)  # (B, L, H*(1 or 2))
        reconstructed = self.reconstructor(lstm_out)
        return reconstructed
