import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import numpy as np
from Forecaster.base_forecaster import BaseForecaster
from torch.utils.data import DataLoader, TensorDataset
from Utils.tools import adjust_learning_rate
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # classification
        self.label_len = configs.label_len  # forecast
        self.pred_len = configs.pred_len  # forecast
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class AutoformerForecaster(BaseForecaster):
    def __init__(self, configs):
        super().__init__()

        # 初始化模型
        self.configs = self._update_configs(configs)
        self.model = Model(configs).to(self.configs.device)

        # 优化器与损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()


    def _update_configs(self, configs):
        """根据传入的参数更新配置，传入的配置会覆盖默认配置"""

        # 定义默认配置
        default_configs = {
            'task_name': 'long_term_forecast',
            'device': 'cpu',
            'logger': None,
            'seq_len': 192,
            'label_len': 96,
            'pred_len': 96,
            'output_attention': False,
            'moving_avg': 25,
            'enc_in': 25,  # 特征维度
            'dec_in': 25,
            'c_out': 1,
            'd_model': 128,
            'embed': 'timeF',
            'freq': 'h',
            'dropout': 0.1,
            'factor': 3,
            'n_heads': 8,
            'd_ff': 128,
            'activation': 'gelu',
            'e_layers': 2,
            'd_layers': 1,
            'learning_rate': 5e-4,
            'batch_size': 512,
            'epochs': 100,
            'lradj': 'type1',
        }

        # 合并配置：传入的配置覆盖默认配置
        merged_configs = {**default_configs, **configs}

        # 转换为 SimpleNamespace，支持点号访问 (cfg.seq_len)
        return SimpleNamespace(**merged_configs)

    def fit(self, data: pd.DataFrame):
        # 1. 维度自适应
        data_scaled = self.scaler.fit_transform(data.values)
        data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)

        feat_dim = data.shape[1]
        self.configs.enc_in = feat_dim
        self.configs.dec_in = feat_dim
        self.configs.c_out = feat_dim

        # 重建模型
        self.model = Model(self.configs).to(self.configs.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)

        # 2. 数据准备
        # split_index = int(len(data) * 0.8)
        split_index = len(data)
        train_data = data.iloc[:split_index]
        val_data = data.iloc[split_index:]

        train_loader = self._create_windows(train_data, shuffle=True)
        # val_loader = self._create_windows(val_data, shuffle=False)

        patience = 5
        best_val_loss = float('inf')
        counter = 0

        # 3. 训练循环
        self.configs.logger.info(f"Start training: Train size={len(train_data)}, Val size={len(val_data)}")

        for epoch in range(self.configs.epochs):
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.configs.device)
                batch_y = batch_y.to(self.configs.device)

                self.optimizer.zero_grad()

                # === 构建 Decoder Input ===
                # 目标结构: [Label (已知) | Placeholder (未知)]
                # 1. 取出 label 部分 (batch_y 的前 label_len)
                label_part = batch_y[:, :self.configs.label_len, :]
                # 2. 创建占位符 (全0) 用于 pred_len
                zeros_part = torch.zeros(
                    [batch_y.shape[0], self.configs.pred_len, batch_y.shape[2]],
                    device=self.configs.device
                )
                # 3. 拼接
                dec_inp = torch.cat([label_part, zeros_part], dim=1)

                # === Forward ===
                # Autoformer forward 签名: x_enc, x_mark_enc, x_dec, x_mark_dec
                # 这里省略 mark (时间戳 embedding)，传入 None
                outputs = self.model(batch_x, None, dec_inp, None)

                # === Loss 计算 ===
                # 模型输出 shape: [Batch, label_len + pred_len, Features]
                # 我们只计算后 pred_len 部分的 loss
                f_dim = -1 if self.configs.c_out == 1 else 0  # 如果是单变量只看最后一列

                pred_out = outputs[:, -self.configs.pred_len:, f_dim:]
                true_y = batch_y[:, -self.configs.pred_len:, f_dim:]

                loss = self.criterion(pred_out, true_y)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            # self.model.eval()
            # val_loss = []
            # with torch.no_grad():
            #     for i, (batch_x, batch_y) in enumerate(val_loader):
            #         batch_x = batch_x.to(self.configs.device)
            #         batch_y = batch_y.to(self.configs.device)
            #
            #         # === 构建 Decoder Input ===
            #         label_part = batch_y[:, :self.configs.label_len, :]
            #         zeros_part = torch.zeros(
            #             [batch_y.shape[0], self.configs.pred_len, batch_y.shape[2]],
            #             device=self.configs.device
            #         )
            #         dec_inp = torch.cat([label_part, zeros_part], dim=1)
            #
            #         # === Forward ===
            #         outputs = self.model(batch_x, None, dec_inp, None)
            #
            #         f_dim = -1 if self.configs.c_out == 1 else 0
            #         pred_out = outputs[:, -self.configs.pred_len:, f_dim:]
            #         true_y = batch_y[:, -self.configs.pred_len:, f_dim:]
            #
            #         loss = self.criterion(pred_out, true_y)
            #         val_loss.append(loss.item())
            train_loss_avg = np.mean(train_loss)
            # val_loss_avg = np.mean(val_loss)
            val_loss_avg = train_loss_avg  # 临时代替

            self.configs.logger.info(
                f"Epoch {epoch + 1}/{self.configs.epochs} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                counter = 0
                self.configs.learning_rate = 1e-4
            else:
                counter += 1
                adjust_learning_rate(self.optimizer, epoch + 1, self.configs)
                if counter >= patience:
                    self.configs.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} (Val loss did not improve for {patience} epochs)")
                    break
            # adjust_learning_rate(self.optimizer, epoch + 1, self.configs)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        对测试序列进行滑动窗口切分，并行进行多步预测，最后拼接结果并求和。
        逻辑：
        1. 按步长 pred_len 滑动截取 seq_len 长度的输入。
        2. 批量输入模型，预测未来的 pred_len 长度。
        3. 将所有预测片段拼接。
        4. 前 seq_len 部分无法预测，用 0 填充。
        5. 对特征维度求和，输出单变量 Series。
        """
        data_scaled = self.scaler.transform(data.values)
        data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)
        self.model.eval()

        seq_len = self.configs.seq_len
        label_len = self.configs.label_len
        pred_len = self.configs.pred_len

        # 1. 准备数据窗口
        # 我们使用 stride = pred_len，保证预测结果是连续且不重叠的
        # 窗口 i 的输入范围: [i, i + seq_len]
        # 窗口 i 的预测范围: [i + seq_len, i + seq_len + pred_len]
        raw_data = data.values
        windows = []

        # 能够完整构建输入窗口的最大起始索引
        limit = len(raw_data) - seq_len

        # 生成切片索引: 0, 24, 48... (假设 pred_len=24)
        for i in range(0, limit + 1, pred_len):
            _x = raw_data[i: i + seq_len, :]
            windows.append(_x)

        if not windows:
            # 数据太短，无法进行任何预测，返回全0
            return pd.Series(0, index=data.index)

        # 转换为 Tensor 并创建 DataLoader 实现并行(Batch)预测
        x_tensor = torch.FloatTensor(np.array(windows))
        # 仅用于推理，batch_size 可设大一点
        batch_size = self.configs.batch_size * 2
        # shuffle=False 非常重要，保证拼接顺序正确
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        prediction_chunks = []

        # 2. 并行推理
        with torch.no_grad():
            for i, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.configs.device)

                # === 构建 Decoder Input ===
                # 取 batch_x 的最后 label_len 作为引导
                batch_label = batch_x[:, -label_len:, :]
                # 占位符
                zeros_part = torch.zeros(
                    [batch_x.shape[0], pred_len, batch_x.shape[2]],
                    device=self.configs.device
                )
                dec_inp = torch.cat([batch_label, zeros_part], dim=1)

                # === Forward ===
                # Autoformer 输出 shape: [Batch, label_len + pred_len, Features]
                outputs = self.model(batch_x, None, dec_inp, None)

                # 只取后 pred_len (未来的预测值)
                f_dim = -1 if self.configs.c_out == 1 else 0
                pred_out = outputs[:, -pred_len:, f_dim:]  # [Batch, pred_len, Features]

                prediction_chunks.append(pred_out.cpu().numpy())

        # 3. 结果拼接与后处理
        # 将 list of batch arrays -> (N_windows * pred_len, Features)
        all_preds = np.concatenate(prediction_chunks, axis=0)  # [Total_Pred_Rows, Features]
        # 展平 batch 和 time 维度: (Batch * pred_len, Features)
        all_preds = all_preds.reshape(-1, all_preds.shape[-1])

        # === 关键步骤：特征维度求和 ===
        # 形状变为 (Total_Length, )
        summed_preds = all_preds.sum(axis=1)

        # 4. 对齐原数据 (Padding)
        # 预测是从第 seq_len 个点开始的 (因为需要前 seq_len 个点作为历史)
        # 所以前 seq_len 个位置填 0
        padding = np.zeros(seq_len)
        final_values = np.concatenate([padding, summed_preds])

        # 5. 截断或填充以匹配原始 DataFrame 的长度
        original_len = len(data)
        current_len = len(final_values)

        if current_len < original_len:
            # 如果滑窗没滑到最后（比如数据末尾不够凑齐一个 pred_len），补0
            tail_pad = np.zeros(original_len - current_len)
            final_values = np.concatenate([final_values, tail_pad])
        elif current_len > original_len:
            # 这种情况通常不会发生，除非 pred_len设置逻辑有溢出，安全起见截断
            final_values = final_values[:original_len]

        # 6. 返回 Series
        return pd.Series(final_values, index=data.index)

    def _create_windows(self, data: pd.DataFrame, shuffle=False):
        """
        构建训练数据的滑动窗口
        X: [t : t + seq_len]
        Y: [t + seq_len - label_len : t + seq_len + pred_len]
        """
        raw_data = data.values
        seq_len = self.configs.seq_len
        label_len = self.configs.label_len
        pred_len = self.configs.pred_len

        x_data = []
        y_data = []

        # 确保数据够长
        total_len = len(raw_data)
        if total_len < seq_len + pred_len:
            raise ValueError(f"Data length {total_len} is too short for seq_len={seq_len} + pred_len={pred_len}")

        # 滑动窗口
        for i in range(total_len - seq_len - pred_len + 1):
            # Encoder Input
            _x = raw_data[i: i + seq_len]

            # Ground Truth for Decoder (包含 label 和 prediction 部分)
            # Decoder 需要 label_len 的历史 + pred_len 的未来
            y_start = i + seq_len - label_len
            y_end = i + seq_len + pred_len
            _y = raw_data[y_start: y_end]

            x_data.append(_x)
            y_data.append(_y)

        x_data = torch.FloatTensor(np.array(x_data))
        y_data = torch.FloatTensor(np.array(y_data))

        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=self.configs.batch_size, shuffle=shuffle)
        return dataloader
