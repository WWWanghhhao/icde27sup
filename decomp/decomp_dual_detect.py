import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler, RobustScaler
from types import SimpleNamespace
import sys, os, json
import argparse, math
import torch.nn.functional as F
import time
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

from Valuator.main import metrics2
from Utils.tools import provide_data_loader, get_logger, set_seed, reconstruct_from_windows
from Detector.MyDetector import SELSTMModel_v3 as DETECTOR
from Forecaster.DUET import DUET
from Processor.rag import OutputFusionRAG2 as OFRAG


class SeriesDecomp(nn.Module):
    """
    Autoformer-style series decomposition:
    x = trend + seasonal
    trend = moving_avg(x), seasonal = x - trend
    """

    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        # x: [B, T, C]
        x_t = x.transpose(1, 2)  # [B, C, T]
        trend = self.avg_pool(x_t).transpose(1, 2)  # [B, T, C]
        seasonal = x - trend
        return seasonal, trend


class Predictor(nn.Module):
    """
    基于 LSTM 的多变量时间序列预测器

    参数:
        seq_len: 输入历史序列长度
        pred_len: 预测未来序列长度
        channels: 变量（通道）数
        hidden_size: LSTM 隐藏层维度 (默认 128)
        num_layers: LSTM 层数 (默认 2)
        dropout: 层间 dropout 比率 (默认 0.0)
        individual: 是否对每个通道独立建模 (暂未实现, 保留接口)
    """

    def __init__(self, seq_len, pred_len, channels, hidden_size=128, num_layers=3, dropout=0.0, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.individual = individual  # 保留但未实现独立的LSTM per channel

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.regressor = nn.Linear(hidden_size, pred_len * channels)

    def forward(self, x):
        """
        x: [B, seq_len, channels]
        返回: [B, pred_len, channels]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        pred_flat = self.regressor(last_hidden)
        pred = pred_flat.view(-1, self.pred_len, self.channels)
        return pred

"""
class Predictor(nn.Module):
    def __init__(self, seq_len, pred_len, channels, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual

        if self.individual:
            self.linear1 = nn.ModuleList()
            self.linear2 = nn.ModuleList()
            for i in range(self.channels):
                lin1 = nn.Linear(self.seq_len, self.pred_len)
                lin2 = nn.Linear(self.seq_len, self.pred_len)
                lin1.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                lin2.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.linear1.append(lin1)
                self.linear2.append(lin2)
        else:
            self.linear1 = nn.Linear(self.seq_len, self.pred_len)
            self.linear2 = nn.Linear(self.seq_len, self.pred_len)
            self.linear1.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.linear2.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x: [B, T, C] -> 转换为 [B, C, T] 以便线性层作用于时间维度
        x = x.permute(0, 2, 1)  # [B, C, T]

        if self.individual:
            out1 = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype).to(x.device)
            out2 = torch.zeros_like(out1)
            for i in range(self.channels):
                out1[:, i, :] = self.linear1[i](x[:, i, :])
                out2[:, i, :] = self.linear2[i](x[:, i, :])
        else:
            out1 = self.linear1(x)   # [B, C, pred_len]
            out2 = self.linear2(x)   # [B, C, pred_len]

        y = out1 + out2
        return y.permute(0, 2, 1)    # 恢复为 [B, pred_len, C]
"""

def parse_args():
    parser = argparse.ArgumentParser(description="training")

    # train_args
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--single", action="store_true", help="use single feature")
    parser.add_argument("--batch_size", type=int, default=64)

    # det_args (CATCH/Detector)
    parser.add_argument("--det_epochs", type=int, default=3)
    parser.add_argument("--det_lr", type=float, default=1e-2)

    # duet_args (Forecaster)
    parser.add_argument("--fst_epochs", type=int, default=3)
    parser.add_argument("--fst_lr", type=float, default=1e-2)

    # pred_len list
    parser.add_argument("--seq_len", type=int, default=192)

    parser.add_argument("--pred_lens", type=int, nargs="+", default=[96])

    parser.add_argument("--datasets", type=str, nargs="+", default=["SMAP"], help="dataset names in map keys")
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--individual", action="store_true", help="individual channel linear")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3047])
    parser.add_argument("--mem_size", type=int, default=64)
    parser.add_argument("--attn_temperature", type=float, default=0.5)
    parser.add_argument("--gate_scale_init", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=3)

    return parser.parse_args()


def nearest_power_of_2(n: int) -> int:
    """返回离 n 最近的 2 的整次幂（四舍五入到 2^k）"""
    if n <= 0:
        return 1
    exponent = round(math.log2(n))
    return max(1, int(2 ** exponent))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


args = parse_args()

train_args = {
    "seq_len": args.seq_len,
    "device": args.device,
    "single": False,
}

path1 = f"./logs/lstm/2026/v13_Final/{args.datasets}"
if not os.path.exists(path1):
    os.makedirs(path1)
# s = f'{path1}/{args.pred_lens}.log'
s = f'{path1}/profile.log'
print(s)
logger = get_logger(s)
logger.info(s)
logger.info("=" * 50)
logger.info("Command-line arguments:")
for arg_name, arg_value in vars(args).items():
    logger.info(f"  {arg_name}: {arg_value}")

train_configs = SimpleNamespace(**train_args)

map = {
    'GECCO': (9, '../dataset/GECCO/', 1),
    'MSL': (55, '../dataset/MSL/', 5),
    'SMAP': (25, '../dataset/SMAP/', 2),
    'PSM': (25, '../dataset/PSM/', 5),
    'Genesis': (18, '../dataset/Genesis/', 5),
    'NYC': (3, '../dataset/NYC/', 5),
    'Creditcard': (29, '../dataset/Creditcard/', 5),
    'SMD': (38, '../dataset/SMD/', 5),
}

with open('../configs/config.json', 'r') as f:
    map_args = json.load(f)
with open('../configs/catch_default_config.json', 'r') as f:
    catch_default_args = json.load(f)
with open('../configs/duet_default_config.json', 'r') as f:
    duet_default_args = json.load(f)


def split_train_val(data, val_ratio=0.2):
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def compute_channel_weights(detector1, detector2, decomp, data_loader, seq_len, device, eps=1e-8):
    """
    基于第一阶段训练后的 detector，在训练集上统计每个通道重构误差，得到通道权重。
    误差越大，权重越高。
    """
    detector1.eval()
    detector2.eval()
    decomp.eval()

    channel_err_sum = None
    sample_count = 0  # 用于做均值归一化（按 B * pred_len）

    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)

            season, trend = decomp(batch_data)
            rec_season = detector1(season)
            rec_trend = detector2(trend)

            season_err = torch.abs(rec_season - season)
            trend_err = torch.abs(rec_trend - trend)
            total_err = season_err + trend_err

            # 对 batch 和时间维求和，保留通道维 [C]
            ch_sum = total_err.sum(dim=(0, 1))

            if channel_err_sum is None:
                channel_err_sum = ch_sum
            else:
                channel_err_sum += ch_sum

            sample_count += total_err.shape[0] * total_err.shape[1]

    if channel_err_sum is None:
        return None

    weights = channel_err_sum / max(sample_count, 1)  # [C]
    return weights  # tensor [C], on device





def train_test(train_data, test_data, det_config, duet_config, dataset_name, anomaly_ratio=5):
    device = train_configs.device

    train_df, val_df = split_train_val(train_data, val_ratio=0.2)

    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_df.values)
    scaled_val = scaler.transform(val_df.values)

    data_loader = provide_data_loader(
        scaled_train,
        train_configs.seq_len + train_configs.pred_len,
        det_config.batch_size,
        shuffle=False, num_workers=0
    )
    val_loader = provide_data_loader(
        scaled_val,
        train_configs.seq_len + train_configs.pred_len,
        det_config.batch_size,
        shuffle=False, num_workers=0
    )

    # ===== 动态 kernel_size: (seq_len + pred_len) / 2 =====
    dynamic_kernel = max(3, int(round((train_configs.seq_len + train_configs.pred_len) / 2)))
    if dynamic_kernel % 2 == 0:
        dynamic_kernel += 1
    # dynamic_kernel = 25
    decomp = SeriesDecomp(kernel_size=dynamic_kernel).to(device)

    forecaster = Predictor(train_configs.seq_len, train_configs.pred_len, train_data.shape[1],
                           individual=args.individual).to(device)

    detector1 = DETECTOR(det_config.c_in, train_configs.seq_len + train_configs.pred_len,
                         hidden_size=nearest_power_of_2(train_data.shape[1]) *2, num_layers=det_config.lstm_layers).to(device)
    detector2 = DETECTOR(det_config.c_in, train_configs.seq_len + train_configs.pred_len,
                         hidden_size=nearest_power_of_2(train_data.shape[1] *2), num_layers=det_config.lstm_layers).to(device)

    logger.info('Mem Size: {}'.format(args.mem_size))
    rag_module = OFRAG(seq_len=train_configs.seq_len, pred_len=train_configs.pred_len, feature_dim=train_data.shape[1],
                       d_model=nearest_power_of_2(train_data.shape[1]), top_k=args.topk, max_size=args.mem_size,
                       attn_temperature=args.attn_temperature, gate_scale_init=args.gate_scale_init).to(device)

    # ===== 统计参数量 =====
    # det_params = count_params(detector1) + count_params(detector2)
    # rag_params = count_params(rag_module)
    # fcast_params = count_params(forecaster)
    # total_params = det_params + rag_params + fcast_params
    # logger.info(f"Params - Detector: {det_params:,} | RAG: {rag_params:,} | Forecaster: {fcast_params:,} | Total: {total_params:,}")

    optimizerD1 = torch.optim.Adam(list(detector1.parameters()), lr=det_config.lr)
    optimizerD2 = torch.optim.Adam(list(detector2.parameters()), lr=det_config.lr)
    optimizerF = optim.Adam(list(forecaster.parameters()) + list(rag_module.parameters()), lr=duet_config.lr)

    criterion = torch.nn.MSELoss(reduction='mean')

    # 第一阶段早停机制
    best_loss_stage1 = float('inf')
    best_val_stage1 = float('inf')
    best_model1 = None
    best_model2_stage1 = None
    min_lr_stage1 = 1e-6
    min_delta = 5e-5
    plateau_counter_stage1 = 0

    # 定义保存路径（可根据需要调整）
    save_dir = "./model_pth/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    det1_path = os.path.join(save_dir,
                             f"{dataset_name}{train_configs.pred_len}_detector1_ep{det_config.epochs}_lr{det_config.lr}.pth")
    det2_path = os.path.join(save_dir,
                             f"{dataset_name}{train_configs.pred_len}_detector2_ep{det_config.epochs}_lr{det_config.lr}.pth")

    # 检查模型文件是否已存在
    # if os.path.exists(det1_path) and os.path.exists(det2_path):
    xx = False
    if xx:
        print(f"load {det1_path}, {det2_path}")
        detector1.load_state_dict(torch.load(det1_path))
        detector2.load_state_dict(torch.load(det2_path))
    else:
        for epoch in range(det_config.epochs):
            forecaster.train()
            detector1.train()
            detector2.train()
            rag_module.train()

            t1 = time.time()
            total_loss = 0
            for batch_data, _ in data_loader:
                batch_data = batch_data.to(device)

                optimizerD1.zero_grad()
                optimizerD2.zero_grad()

                season, trend = decomp(batch_data)
                rec_season = detector1(season)
                rec_trend = detector2(trend)

                loss = criterion(rec_season, season) + criterion(rec_trend, trend)

                loss.backward()
                optimizerD1.step()
                optimizerD2.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)

            detector1.eval()
            detector2.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    batch_data = batch_data.to(device)
                    season, trend = decomp(batch_data)
                    rec_season = detector1(season)
                    rec_trend = detector2(trend)
                    loss = criterion(rec_season, season) + criterion(rec_trend, trend)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_loss
            logger.info(
                f"Stage 1 - Epoch {epoch + 1}/{det_config.epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            t2 = time.time()
            print(f"Time1 {t1-t2}")
            if best_val_stage1 - val_loss < min_delta:
                plateau_counter_stage1 += 1
            else:
                plateau_counter_stage1 = 0

            if val_loss < best_val_stage1:
                best_val_stage1 = val_loss

            if avg_loss < best_loss_stage1:
                best_loss_stage1 = avg_loss
                best_model1 = detector1.state_dict()
                best_model2_stage1 = detector2.state_dict()
            else:
                old_lr1 = optimizerD1.param_groups[0]['lr']
                if old_lr1 > min_lr_stage1:
                    new_lr1 = old_lr1 * 0.1
                    for param_group in optimizerD1.param_groups:
                        param_group['lr'] = new_lr1
                    for param_group in optimizerD2.param_groups:
                        param_group['lr'] = new_lr1

            if plateau_counter_stage1 >= 3:
                logger.info(f"Stage 1: Early stopping triggered after {epoch + 1} epochs (val plateau)")
                break

        if best_model1 is not None:
            detector1.load_state_dict(best_model1)
            detector2.load_state_dict(best_model2_stage1)
            # torch.save(best_model1, det1_path)
            # torch.save(best_model2_stage1, det2_path)

    # ===== 新增：根据第一阶段重构误差计算通道权重 =====
    channel_weights = compute_channel_weights(
        detector1=detector1,
        detector2=detector2,
        decomp=decomp,
        data_loader=data_loader,
        seq_len=train_configs.seq_len,
        device=device
    )
    if channel_weights is None:
        channel_weights = torch.ones(train_data.shape[1], device=device) / train_data.shape[1]

    for param in detector1.parameters():
        param.requires_grad = False
    for param in detector2.parameters():
        param.requires_grad = False

    # 第二阶段早停机制
    best_loss_stage2 = float('inf')
    best_val_stage2 = float('inf')
    best_forecaster = None
    plateau_counter_stage2 = 0

    for epoch in range(duet_config.epochs):
        forecaster.train()
        rag_module.train()
        t1 = time.time()
        total_loss = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)

            optimizerF.zero_grad()

            season, trend = decomp(batch_data)
            rec_season = detector1(season)
            rec_trend = detector2(trend)

            season_key, _ = detector1.forward_with_multiscale(season)
            trend_key, _ = detector2.forward_with_multiscale(trend)
            season_val = torch.abs(rec_season - season)
            trend_val = torch.abs(rec_trend - trend)

            max_val = torch.max(season_val, trend_val)

            key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()
            val = max_val[:, train_configs.seq_len:, :].detach()

            rag_module.update_memory(key, val)

            pred_val_raw = forecaster(key)
            pred_val, cl_loss = rag_module(key, pred_val_raw)
            loss = criterion(pred_val, val) + cl_loss

            loss.backward()
            optimizerF.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        forecaster.eval()
        rag_module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)

                season, trend = decomp(batch_data)
                rec_season = detector1(season)
                rec_trend = detector2(trend)
                season_key, _ = detector1.forward_with_multiscale(season)
                trend_key, _ = detector2.forward_with_multiscale(trend)
                season_val = torch.abs(rec_season - season)
                trend_val = torch.abs(rec_trend - trend)
                max_val = torch.max(season_val, trend_val)
                key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()
                val = max_val[:, train_configs.seq_len:, :].detach()

                rag_module.update_memory(key, val)

                pred_val_raw = forecaster(key)
                pred_val, cl_loss = rag_module(key, pred_val_raw)
                loss = criterion(pred_val, val) + cl_loss

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_loss
        logger.info(f"Stage 2 - Epoch {epoch + 1}/{duet_config.epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        t2 = time.time()
        print(f"Time2 {t1-t2}")
        if best_val_stage2 - val_loss < min_delta:
            plateau_counter_stage2 += 1
        else:
            plateau_counter_stage2 = 0

        if val_loss < best_val_stage2:
            best_val_stage2 = val_loss

        if avg_loss < best_loss_stage2:
            best_loss_stage2 = avg_loss
            best_forecaster = forecaster.state_dict()
        else:
            old_lr2 = optimizerF.param_groups[0]['lr']
            if old_lr2 > min_lr_stage1:
                new_lr2 = old_lr2 * 0.1
                for param_group in optimizerF.param_groups:
                    param_group['lr'] = new_lr2

        if plateau_counter_stage2 >= 3:
            logger.info(f"Stage 2: Early stopping triggered after {epoch + 1} epochs (val plateau)")
            if best_forecaster is not None:
                forecaster.load_state_dict(best_forecaster)
            break

    if best_forecaster is not None:
        forecaster.load_state_dict(best_forecaster)

    scaled_test_data = scaler.transform(test_data.values)
    test_loader = provide_data_loader(
        scaled_test_data,
        train_configs.seq_len + train_configs.pred_len,
        det_config.batch_size,
        shuffle=False, num_workers=0
    )

    forecaster.eval()
    detector2.eval()
    detector1.eval()
    rag_module.eval()
    rag_module.memory_keys = None
    rag_module.memory_values = None

    # 先用训练集构建 memory
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            season, trend = decomp(batch_data)
            rec_season = detector1(season)
            rec_trend = detector2(trend)
            season_key, _ = detector1.forward_with_multiscale(season)
            trend_key, _ = detector2.forward_with_multiscale(trend)
            season_val = torch.abs(rec_season - season)
            trend_val = torch.abs(rec_trend - trend)
            max_val = torch.max(season_val, trend_val)
            key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()
            val = max_val[:, train_configs.seq_len:, :].detach()
            rag_module.update_memory(key, val)

    attn_test = []
    # total_time = 0.0
    # total_samples = 0


    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.to(device)
            batch_data[:, train_configs.seq_len:, :] = 0
            season, trend = decomp(batch_data)
            season_key, _ = detector1.forward_with_multiscale(season)
            trend_key, _ = detector2.forward_with_multiscale(trend)

            key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()

            pred_val_raw = forecaster(key)
            pred_val, cl_loss = rag_module(key, pred_val_raw)  # [B, pred_len, C]

            anomaly_score = torch.sum(torch.abs(pred_val) * channel_weights.view(1, 1, -1), dim=2)
            pred_score = anomaly_score.mean(dim=1)  # [B]
            # pred_score = anomaly_score[:, 0]  # [B] 取预测窗口第一个时间步
            attn_test.append(pred_score.cpu().numpy())

    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         record_shapes=True,
    #         with_flops=True
    # ) as prof:
    #     with torch.no_grad():
    #         for batch_data, _ in test_loader:
    #             total_samples += batch_data.size(0)
    #
    #             batch_data = batch_data.to(device)
    #             batch_data[:, train_configs.seq_len:, :] = 0
    #
    #             season, trend = decomp(batch_data)
    #             season_key, _ = detector1.forward_with_multiscale(season)
    #             trend_key, _ = detector2.forward_with_multiscale(trend)
    #
    #             key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()
    #             pred_val_raw = forecaster(key)
    #             pred_val, _ = rag_module(key, pred_val_raw)
    #
    # # 统计推理总 FLOPs
    # total_mflops = 0.0
    # for evt in prof.key_averages():
    #     if evt.flops is not None:
    #         total_mflops += evt.flops / 1e6  # 转成 MFLOPs
    #
    # total_gflops = total_mflops / 1000.0
    # print(f"Test inference total FLOPs: {total_gflops:.4f} GFLOPs")
    # exit()


    attn_test = np.concatenate(attn_test)
    # logger.info(f"Test Inference Time: {total_time:.4f}s | Total Samples: {total_samples}")

    # 训练分数
    attn_train = []
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            batch_data[:, train_configs.seq_len:, :] = 0
            season, trend = decomp(batch_data)
            season_key, _ = detector1.forward_with_multiscale(season)
            trend_key, _ = detector2.forward_with_multiscale(trend)

            key = (season_key + trend_key)[:, :train_configs.seq_len, :].detach()

            pred_val_raw = forecaster(key)
            pred_val, cl_loss = rag_module(key, pred_val_raw)

            anomaly_score = torch.sum(torch.abs(pred_val) * channel_weights.view(1, 1, -1), dim=2)
            pred_score = anomaly_score.mean(dim=1)
            attn_train.append(pred_score.cpu().numpy())
    attn_train = np.concatenate(attn_train)

    train_energy = np.array(attn_train)
    test_energy = np.array(attn_test)

    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)

    return attn_test, thresh


if __name__ == '__main__':
    pred_lens = args.pred_lens
    seeds = args.seeds

    selected_datasets = {k: v for k, v in map.items() if k in args.datasets}

    for p in pred_lens:
        train_configs.pred_len = p

        for dataset_name, (feature_dim, dataset_path, anomaly_ratio) in selected_datasets.items():
            logger.info(
                f"Processing dataset: {dataset_name}, seq_len: {train_configs.seq_len}, pred_len: {train_configs.pred_len}")

            train_filepath = f"{dataset_path}train.csv"
            test_filepath = f"{dataset_path}test.csv"

            train_data = pd.read_csv(train_filepath)
            test_data = pd.read_csv(test_filepath)
            test_label = test_data['label']

            feature_cols = [f'col_{i}' for i in range(feature_dim)]

            # 训练集缺失值填充
            train_missing = train_data[feature_cols].isnull().sum().sum()
            train_medians = train_data[feature_cols].median()
            train_data[feature_cols] = train_data[feature_cols].fillna(train_medians)
            # 测试集也做同样填充（用训练集统计量）
            test_data[feature_cols] = test_data[feature_cols].fillna(train_medians)

            det_cfg_dict = catch_default_args.copy()
            det_cfg_dict.update(map_args[dataset_name])
            det_cfg_dict['seq_len'] = train_configs.seq_len + train_configs.pred_len
            det_cfg_dict['device'] = train_configs.device
            det_cfg_dict['c_in'] = feature_dim

            det_cfg_dict['epochs'] = args.det_epochs
            det_cfg_dict['batch_size'] = args.batch_size
            det_cfg_dict['lr'] = args.det_lr
            det_cfg_dict['lstm_layers'] = 3
            if dataset_name in ['Genesis', 'NYC']:
                det_cfg_dict['lstm_layers'] = 2
            det_config = SimpleNamespace(**det_cfg_dict)

            duet_cfg_dict = duet_default_args.copy()
            duet_cfg_dict['enc_in'] = feature_dim
            duet_cfg_dict['pred_len'] = train_configs.pred_len
            duet_cfg_dict['seq_len'] = train_configs.seq_len
            duet_cfg_dict['device'] = train_configs.device
            duet_cfg_dict['epochs'] = args.fst_epochs
            duet_cfg_dict['lr'] = args.fst_lr
            duet_config = SimpleNamespace(**duet_cfg_dict)

            for seed in seeds:
                logger.info(f"\nRunning with seed: {seed}")
                set_seed(seed)

                scores, thresh = train_test(
                    train_data[feature_cols],
                    test_data[feature_cols],
                    det_config,
                    duet_config,
                    dataset_name,
                    anomaly_ratio=anomaly_ratio
                )
                print(scores.min(), scores.max(), scores.std())
                # exit()

                label = test_label.values
                labels = label[train_configs.seq_len:]

                metrics2(scores, labels[:len(scores)], thresh, logger)

            logger.info(f"{dataset_name} finished.\n-----------------------------------\n")