import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import numpy as np
from Detector.base_detector import BaseDetector
from torch.utils.data import DataLoader, TensorDataset, Dataset
from Utils.tools import adjust_learning_rate
from sklearn.preprocessing import StandardScaler



class LazySlidingWindowDataset(Dataset):
    def __init__(self, data_array: np.ndarray, seq_len: int):
        """
        只存储原始数据的引用，不在初始化时进行切片
        """
        # 强制转换为 float32 节省一半内存
        self.data = torch.from_numpy(data_array.astype(np.float32))
        self.seq_len = seq_len
        self.n_samples = max(0, len(data_array) - seq_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 实时切片，内存占用极低
        window = self.data[idx : idx + self.seq_len]
        return window, 0 # 0是dummy target


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


class AutoformerDetector(BaseDetector):
    def __init__(self, **kwargs):
        """
        kwargs 参数示例:
        seq_len=96, d_model=512, d_ff=2048, n_heads=8, e_layers=2,
        dropout=0.05, learning_rate=0.0001, batch_size=32, epochs=10
        """
        super().__init__(**kwargs)
        self.args = self._parse_configs(kwargs)

        # 初始化模型
        self.model = Model(self.args).to(self.args.device)

        # 优化器与损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()

    def _parse_configs(self, kwargs):
        """创建一个简单的配置对象来模拟 argparse.Namespace"""

        class Config:
            pass

        cfg = Config()
        # 默认参数设置
        cfg.task_name = 'anomaly_detection'
        cfg.device = kwargs.get('device', 'cpu')
        cfg.logger = kwargs.get('logger', None)
        cfg.seq_len = kwargs.get('seq_len', 96 * 2)
        cfg.label_len = 0  # AD通常不需要
        cfg.pred_len = 0  # AD通常不需要
        cfg.output_attention = False
        cfg.moving_avg = kwargs.get('moving_avg', 25)
        cfg.enc_in = kwargs.get('enc_in', 25)  # 特征维度
        cfg.dec_in = kwargs.get('dec_in', 7)
        # cfg.c_out = kwargs.get('input_size', 1)  # 输出维度应与输入一致
        cfg.c_out = kwargs.get('enc_in', 25)
        cfg.d_model = kwargs.get('d_model', 128)
        cfg.embed = 'timeF'
        cfg.freq = 'h'
        cfg.dropout = kwargs.get('dropout', 0.1)
        cfg.factor = 1
        cfg.n_heads = kwargs.get('n_heads', 8)
        cfg.d_ff = kwargs.get('d_ff', 128)
        cfg.activation = 'gelu'
        cfg.e_layers = kwargs.get('e_layers', 3)
        cfg.d_layers = kwargs.get('d_layers', 2)
        cfg.learning_rate = kwargs.get('learning_rate', 5e-4)
        cfg.batch_size = kwargs.get('batch_size', 512)
        cfg.epochs = kwargs.get('epochs', 100)
        cfg.lradj = kwargs.get('lradj', 'type1')
        cfg.single_output = kwargs.get('single_output', False)
        return cfg

    def _create_windows(self, data: pd.DataFrame, shuffle=False):
        """
        将 2D DataFrame (T, F) 转换为 3D Tensor (B, Seq_Len, F)
        使用滑动窗口机制
        """
        # 转换为numpy
        raw_data = data.values
        seq_len = self.args.seq_len

        x_data = []
        L = len(raw_data)

        # 滑动窗口切片
        for i in range(L - seq_len + 1):
            # 取长度为 seq_len 的窗口
            _x = raw_data[i: i + seq_len, :]
            x_data.append(_x)

        x_data = np.array(x_data)  # shape: (N_windows, seq_len, n_features)

        tensor_x = torch.FloatTensor(x_data)

        # 在 AD 任务中，target 通常就是 input 自身 (Reconstruction)
        dataset = TensorDataset(tensor_x, tensor_x)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle)

        return dataloader

    # def fit(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     训练模型并返回训练数据的异常分数
    #     """

    #     data_scaled = self.scaler.fit_transform(data.values)
    #     data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)

    #     # 1. 更新输入维度配置
    #     self.args.enc_in = data.shape[1]
    #     self.args.c_out = data.shape[1]

    #     # 重新初始化模型以匹配特征维度
    #     self.model = Model(self.args).to(self.args.device)
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    #     # 2. 划分训练集与验证集 (前80%训练，后20%验证)
    #     # 注意：时间序列数据必须按顺序划分，不能随机打乱
    #     # split_index = int(len(data) * 0.8)
    #     split_index = len(data)
    #     train_df = data.iloc[:split_index]
    #     val_df = data.iloc[split_index:]

    #     # 这里的 shuffle=True 仅针对训练集内部的窗口进行打乱，不影响时序划分
    #     train_loader = self._create_windows(train_df, shuffle=True)
    #     # val_loader = self._create_windows(val_df, shuffle=False)

    #     # 3. 定义早停参数
    #     patience = 5  # 容忍多少个epoch不下降
    #     counter = 0  # 当前计数器
    #     best_val_loss = float('inf')  # 记录最佳验证集损失

    #     self.args.logger.info(f"Start training: Train size={len(train_df)}, Val size={len(val_df)}")

    #     # 4. 训练循环
    #     for epoch in range(self.args.epochs):
    #         # --- 训练阶段 ---
    #         self.model.train()
    #         train_losses = []
    #         for batch_x, _ in train_loader:
    #             batch_x = batch_x.to(self.args.device)
    #             self.optimizer.zero_grad()

    #             outputs = self.model(batch_x)
    #             loss = self.criterion(outputs, batch_x)

    #             loss.backward()
    #             self.optimizer.step()
    #             train_losses.append(loss.item())

    #         # --- 验证阶段 ---
    #         # self.model.eval()
    #         # val_losses = []
    #         # with torch.no_grad():
    #         #     for batch_x, _ in val_loader:
    #         #         batch_x = batch_x.to(self.args.device)
    #         #         outputs = self.model(batch_x)
    #         #         loss = self.criterion(outputs, batch_x)
    #         #         val_losses.append(loss.item())

    #         train_loss_avg = np.mean(train_losses)
    #         # val_loss_avg = np.mean(val_losses)
    #         val_loss_avg = train_loss_avg
    #         self.args.logger.info(
    #             f"Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")

    #         if val_loss_avg < best_val_loss:
    #             best_val_loss = val_loss_avg
    #             counter = 0  # 只要有提升，计数器归零
    #         else:
    #             counter += 1
    #             adjust_learning_rate(self.optimizer, epoch + 1, self.args)
    #             if counter >= patience:
    #                 self.args.logger.info(
    #                     f"Early stopping triggered at epoch {epoch + 1} (Val loss did not improve for {patience} epochs)")
    #                 break  # 直接跳出循环，停止训练

    #         # adjust_learning_rate(self.optimizer, epoch + 1, self.args)

    #     # 5. 返回全量数据的异常分数
    #     return self.predict(data)

    # def predict(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     对数据进行推理，计算重构误差作为异常分数
    #     """
    #     data_scaled = self.scaler.transform(data.values)
    #     data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)
    #     self.model.eval()
    #
    #     # 注意：为了保持输出行数与输入一致，我们需要处理滑动窗口带来的边界问题。
    #     # 策略：通常只取每个窗口的最后一个点，或者对重叠部分取平均。
    #     # 简单起见，这里我们计算每个窗口的重构误差，并补全前 seq_len-1 个数据为 0 或均值。
    #
    #     data_loader = self._create_windows(data, shuffle=False)
    #     preds = []
    #     print('here2')
    #     with torch.no_grad():
    #         print('here3')
    #         for i, (batch_x, _) in enumerate(data_loader):
    #             batch_x = batch_x.to(self.args.device)
    #             outputs = self.model(batch_x)
    #
    #             # 计算重构误差 (B, Seq_Len, Features)
    #             # error = (input - reconstruction) ^ 2
    #             error = torch.pow(batch_x - outputs, 2)
    #
    #             # 这里我们需要将窗口误差映射回原时间序列。
    #             # 为了符合 (num_samples, num_features) 的输出格式，
    #             # 最简单的方法是取每个窗口的最后一个时间步的误差作为该时刻的误差。
    #             # (Batch, Seq, Feat) -> (Batch, Feat) 取 [:, -1, :]
    #             print('here4')
    #             last_step_error = error[:, -1, :]
    #             if self.args.single_output:
    #                 score_sum = last_step_error.sum(dim=-1).cpu().numpy()
    #             else:
    #                 score_sum = last_step_error.cpu().numpy()
    #             preds.append(score_sum)
    #     print('here5')
    #     preds = np.concatenate(preds, axis=0)
    #
    #     # 处理滑动窗口造成的头部缺失 (前 seq_len - 1 个点)
    #     # 我们可以用第一个预测值填充，或者填0
    #     pad_len = len(data) - len(preds)
    #     if pad_len > 0:
    #         if self.args.single_output:
    #             padding = np.zeros(pad_len)
    #         else:
    #             padding = np.zeros((pad_len, data.shape[1]))
    #         final_scores = np.concatenate([padding, preds], axis=0)
    #     else:
    #         final_scores = preds
    #
    #     # 构造返回的 DataFrame，保持列名和索引一致
    #     score_df = pd.DataFrame(final_scores, columns= ['col_0'] if self.args.single_output else data.columns, index=data.index)
    #
    #     return score_df

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        内存优化版：对数据进行推理，计算重构误差 (Overlapping Average)。
        """
        self.model.eval()
        data_scaled = self.scaler.transform(data.values)
        data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)

        # === 优化 1: 准备数据 (使用 numpy array 引用，而非复制) ===
        # 转换为 float32 以节省内存
        data_values = data.values.astype(np.float32)
        seq_len = self.args.seq_len

        # 使用自定义的惰性 Dataset
        # dataset = LazySlidingWindowDataset(data_values, seq_len)
        # num_workers=0 避免多进程带来的额外内存开销，大内存场景建议设为 0 或 1
        # data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        data_loader = self._create_windows_lazy(data,shuffle=False)

        # === 优化 2: 初始化累加器 (使用 float32) ===
        total_len = len(data)
        num_features = data.shape[1]

        # 如果数据量极大 (例如 >1亿行)，可以考虑使用 np.memmap 存放在磁盘上
        # score_accumulator = np.memmap('score.dat', dtype='float32', mode='w+', shape=(total_len, num_features))
        score_accumulator = np.zeros((total_len, num_features), dtype=np.float32)
        count_accumulator = np.zeros((total_len, num_features), dtype=np.float32)

        current_idx = 0

        self.args.logger.info(f'Starting prediction on {total_len} samples using optimized memory strategy...')

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(data_loader):
                batch_x = batch_x.to(self.args.device)

                # 模型推理
                outputs = self.model(batch_x)

                # 计算误差 (B, Seq, F)
                # 注意：不要保留梯度，不要保留整个图
                error = torch.pow(batch_x - outputs, 2)

                # 立即转回 CPU 并转为 numpy，释放 GPU 显存
                error_np = error.cpu().numpy()

                # 获取当前 batch 的实际大小 (最后一个 batch 可能不满)
                batch_size = error_np.shape[0]

                # === 优化 3: 向量化累加 (比 for 循环更快) ===
                # 构造切片索引
                # 我们需要把 (B, Seq, F) 加到 (Total, F) 上
                # 这是一个对角线叠加的问题，为了避免Python循环，可以使用如下逻辑：

                # 方案 A: 如果内存极其紧张，保留原来的双层循环(针对batch内部)，虽然慢但内存最省
                # 方案 B: 使用 numpy 的高级索引 (稍微吃一点内存，但快很多)

                # 这里采用折中方案：只对 batch 维循环，因为 batch_size 通常不大 (32-128)
                for b in range(batch_size):
                    global_start = current_idx + b
                    global_end = global_start + seq_len

                    # 原地累加 (In-place add) 节省内存
                    score_accumulator[global_start: global_end] += error_np[b]
                    count_accumulator[global_start: global_end] += 1

                current_idx += batch_size

                # 显式清理缓存 (可选)
                del error, outputs, batch_x, error_np
                # torch.cuda.empty_cache() # 频繁调用会变慢，仅在极度缺显存时打开

        # 计算平均值
        # 避免除以 0
        count_accumulator[count_accumulator == 0] = 1.0
        avg_scores = score_accumulator / count_accumulator

        # 释放累加器内存
        del score_accumulator, count_accumulator

        # === 构造结果 ===
        if self.args.single_output:
            final_scores = avg_scores.sum(axis=-1)
            columns = ['col_0']
        else:
            final_scores = avg_scores
            columns = data.columns

        score_df = pd.DataFrame(final_scores, columns=columns, index=data.index)

        return score_df


    def _create_windows_lazy(self, data: pd.DataFrame, shuffle=True):
        """
        使用惰性 Dataset 创建 DataLoader，解决内存爆炸问题
        """
        seq_len = self.args.seq_len
        # 获取 numpy 数组
        data_values = data.values
        
        # 使用自定义的 Lazy Dataset
        dataset = LazySlidingWindowDataset(data_values, seq_len)
        
        # num_workers=0 在内存紧张时最安全，避免多进程内存复制
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=shuffle, 
            num_workers=0,
            drop_last=True # 训练时丢弃最后一个不完整的 batch 有助于稳定
        )
        return dataloader

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        训练模型并返回训练数据的异常分数 (内存优化版)
        """
        # === 优化1: 强制 float32 ===
        # 先进行 scaling，然后立即转为 float32
        data_scaled = self.scaler.fit_transform(data.values).astype(np.float32)
        data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

        # 1. 更新输入维度配置
        self.args.enc_in = data.shape[1]
        self.args.c_out = data.shape[1]

        # 重新初始化模型
        self.model = Model(self.args).float().to(self.args.device) # 确保模型也是 float32
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        # 2. 准备训练数据
        # 使用全部数据进行训练 (根据你提供的代码，你似乎用全量数据训练)
        self.args.logger.info(f"Start training on {len(data)} samples...")
        
        # === 优化2: 使用惰性 DataLoader ===
        train_loader = self._create_windows_lazy(data, shuffle=True)
        patience = 5  # 容忍多少个epoch不下降
        counter = 0  # 当前计数器
        best_val_loss = float('inf')  # 记录最佳验证集损失

        # 3. 训练循环
        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []
            
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.to(self.args.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                # 可选：显式释放显存 (如果 GPU 显存也紧张)
                # del batch_x, outputs, loss

            train_loss_avg = np.mean(train_losses)
            if train_loss_avg < best_val_loss:
                best_val_loss = train_loss_avg
                counter = 0  # 只要有提升，计数器归零
                self.args.learning_rate = 1e-4
            else:
                counter += 1
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)
                if counter >= patience:
                    self.args.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} (Train loss did not improve for {patience} epochs)")
                    break  # 直接跳出循环，停止训练

            self.args.logger.info(f"Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss_avg:.6f}")

        return self.predict(data)