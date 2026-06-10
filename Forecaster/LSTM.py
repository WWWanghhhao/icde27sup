import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from .base_forecaster import BaseForecaster
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 修改部分：x shape 的最后一维是特征维度，不再是固定的1
        # x shape: (batch, lookback_window, input_size)
        # out shape: (batch, lookback_window, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # 取序列最后一个时间步的输出进行预测
        # out[:, -1, :] shape: (batch, hidden_size)
        out = out[:, -1, :]

        # prediction shape: (batch, output_size)
        # 修改部分：output_size 应该是 forecast_window
        prediction = self.fc(out)
        return prediction


class LSTMForecaster(BaseForecaster):
    def __init__(self,
                 lookback_window: int = 50,
                 forecast_window: int = 10,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 batch_size: int = 512,
                 feature_num: int = 1,
                 epochs: int = 100,
                 lr: float = 0.001,
                 patience=10,
                 scheduler_step: int = 10,
                 scheduler_gamma: float = 0.5,
                 device: str = 'cpu',
                 single_output=True,
                 logger=None):

        super().__init__()
        self.patience = patience
        self.lookback_window = lookback_window
        self.forecast_window = forecast_window
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.single = single_output

        # 初始化模型
        self.model = LSTMModel(input_size=1 if self.single else feature_num,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               output_size=forecast_window if self.single else self.forecast_window * feature_num).to(
            self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = StandardScaler()
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.logger = logger

    def fit(self, data):
        """
        训练模型
        """

        # 1. 数据标准化
        values = data.values.astype(np.float32)
        scaled_values = self.scaler.fit_transform(values)
        train_size = int(len(scaled_values) * 0.8)
        train_data = scaled_values[:train_size]
        val_data = scaled_values[train_size:]
        # 2. 构建训练集 Batch划分
        X_train, Y_train = self._create_windows(train_data, mode='train')
        X_val, Y_val = self._create_windows(val_data, mode='train')

        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, Y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Start training on {len(dataset)} samples...")

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        # 3. 训练循环
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_Y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                if batch_Y.dim() == 3 and self.single:
                    batch_Y = batch_Y.squeeze(-1)  # Shape: (batch, forecast_window)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                if not self.single:
                    outputs = outputs.view(outputs.shape[0], -1, batch_Y.shape[-1])
                loss = self.criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            self.scheduler.step()
            avg_loss = total_loss / len(dataloader)

            val_total_loss = 0
            self.model.eval()
            with torch.no_grad():
                for val_batch_X, val_batch_Y in val_dataloader:
                    val_batch_X = val_batch_X.to(self.device)
                    val_batch_Y = val_batch_Y.to(self.device)
                    if val_batch_Y.dim() == 3 and self.single:
                        val_batch_Y = val_batch_Y.squeeze(-1)  # Shape: (batch, forecast_window)

                    val_outputs = self.model(val_batch_X)
                    if not self.single:
                        val_outputs = val_outputs.view(val_outputs.shape[0], -1, val_batch_Y.shape[-1])
                    val_loss = self.criterion(val_outputs, val_batch_Y)
                    val_total_loss += val_loss.item()
            val_avg_loss = val_total_loss / len(val_dataloader)
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(dataloader):.4f}, Val Loss: {val_avg_loss:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Best model loaded with loss: {best_val_loss:.6f}")

    def predict(self, data) -> pd.Series:
        """
        预测逻辑：
        1. 将测试序列按步长=forecast_window切分为多个batch
        2. 并行预测
        3. 拼接结果并填充头部的0
        """
        self.model.eval()

        # 1. 数据准备与标准化
        values = data.values.astype(np.float32)
        scaled_values = self.scaler.transform(values)

        # 2. 构建预测用的 Batch 数据
        # 这里生成的所有X用于并行预测未来的片段
        X_pred, _ = self._create_windows(scaled_values, mode='predict')

        if X_pred.shape[0] == 0:
            # 数据长度不足以进行一次预测，返回全0
            return pd.Series(np.zeros(len(data)), index=data.index)

        # 3. 并行预测 (使用 DataLoader 进行 Batch 处理，防止显存溢出)
        dataset = TensorDataset(X_pred)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)  # TensorDataset返回tuple
                outputs = self.model(batch_X)  # Shape: (batch, forecast_window)
                if not self.single:
                    outputs = outputs.view(outputs.shape[0], -1, batch_X.shape[-1])
                all_predictions.append(outputs.cpu().numpy())

        # 4. 拼接预测结果
        # 结果形状: (num_windows, forecast_window) -> Flatten
        if len(all_predictions) > 0:
            if self.single:
                flat_preds = np.concatenate(all_predictions, axis=0).flatten()
            else:
                pred_array = np.concatenate(all_predictions, axis=0)
                if pred_array.ndim == 2:
                    # 如果是二维，假设是单变量输出
                    flat_preds = pred_array.flatten()
                else:
                    # 如果是三维，处理多变量输出
                    # 形状: (num_windows, forecast_window, n_features)
                    # 我们需要重塑为 (num_windows * forecast_window, n_features)
                    flat_preds = pred_array.reshape(-1, pred_array.shape[-1])
        else:
            flat_preds = np.array([])

        # 5. Sklearn 反标准化 (Inverse Transform)
        if self.single:
            flat_preds_reshaped = flat_preds.reshape(-1, 1)
            inv_preds = self.scaler.inverse_transform(flat_preds_reshaped)
            inv_preds = inv_preds.flatten()  # 变回 1D 数组
        else:
            if flat_preds.ndim == 1:
                # 如果是一维，先reshape为二维
                flat_preds_reshaped = flat_preds.reshape(-1, 1)
                inv_preds = self.scaler.inverse_transform(flat_preds_reshaped)
                inv_preds = inv_preds.flatten()
            else:
                # 保持二维形状进行反标准化
                inv_preds = self.scaler.inverse_transform(flat_preds)

        # 6. 填充起始部分的 0 (Lookback部分无预测值)
        # 注意：我们生成的第一个预测对应的时间点是 data[lookback : lookback+forecast]
        # 因此前面需要填充 lookback_window 长度的 0
        if self.single:
            padding = np.zeros(self.lookback_window)
            full_pred_values = np.concatenate([padding, inv_preds])
        else:
            if inv_preds.ndim == 1:
                # 如果是一维，使用一维填充
                padding = np.zeros(self.lookback_window)
                full_pred_values = np.concatenate([padding, inv_preds])
            else:
                # 如果是二维，使用二维填充
                padding = np.zeros((self.lookback_window, inv_preds.shape[1]))
                full_pred_values = np.vstack([padding, inv_preds])

        # 7. 对齐索引
        # 修改部分：根据self.single调整对齐方式
        target_length = len(data)
        current_length = len(full_pred_values)

        if current_length > target_length:
            full_pred_values = full_pred_values[:target_length]
        elif current_length < target_length:
            if self.single or full_pred_values.ndim == 1:
                # 单一维度填充
                pad_end = np.zeros(target_length - current_length)
                full_pred_values = np.concatenate([full_pred_values, pad_end])
            else:
                # 多维度填充
                pad_end = np.zeros((target_length - current_length, full_pred_values.shape[1]))
                full_pred_values = np.vstack([full_pred_values, pad_end])

        # 修改部分：返回Series，对所有特征求和
        if self.single or full_pred_values.ndim == 1:
            # 单一列：直接返回Series
            return pd.Series(full_pred_values, index=data.index)
        else:
            # 多列：对所有特征求和后返回Series
            summed_values = np.sum(full_pred_values, axis=1)
            return pd.Series(summed_values, index=data.index)

    def _create_windows(self, data: np.ndarray, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据模式构建滑动窗口数据
        mode='train': 步长为1，尽可能多地生成训练样本，返回 X, Y
        mode='predict': 步长为 forecast_window，用于非重叠拼接预测，返回 X, None
        """
        X, Y = [], []
        L = self.lookback_window
        F = self.forecast_window
        n_samples = len(data)

        if mode == 'train':
            # 训练模式：滑动步长为1，生成 (X, Y)
            # 保证最后一个窗口有对应的 Y
            for i in range(n_samples - L - F + 1):
                X.append(data[i: i + L])
                Y.append(data[i + L: i + L + F])

            if len(X) == 0:
                raise ValueError("Data too short for the given lookback and forecast windows.")

            return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

        elif mode == 'predict':
            # 预测模式：滑动步长为 forecast_window
            # 这里的目的是生成一系列输入，使得输出刚好能首尾相接
            # Start: 0 -> Input: 0~L -> Output: L~L+F
            # Next:  0+F -> Input: F~L+F -> Output: L+F~L+2F
            for i in range(0, n_samples - L + 1, F):
                X.append(data[i: i + L])

            # 注意：这里只返回X，因为是预测
            if len(X) == 0:
                # 如果数据甚至不够一个lookback，返回空tensor
                return torch.empty(0), None

            return torch.FloatTensor(np.array(X)), None
