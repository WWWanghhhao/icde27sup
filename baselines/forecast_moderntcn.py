import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from types import SimpleNamespace
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

from Valuator.main import metrics2
from Utils.tools import provide_forecasting_data_loader, get_logger, set_seed, reconstruct_from_windows
from Forecaster.Autoformer import Model as Autoformer
from Forecaster.DUET import DUET as Duet
from Forecaster.PatchTST import Model as PatchTST
from Forecaster.DLinear import Model as DLinear
from Forecaster.TimeMixer import Model as TimeMixer
from Detector.ModernTCN_layers.ModernTCN_base import ModernTCN

train_args = {
    "seq_len": 192,
    "pred_len": 128,
    "device": "cuda:0",
    "forecaster": "duet",
}

path1 = f"./logs/f_d/thresh/{train_args['forecaster']}-moderntcn"
os.makedirs(path1, exist_ok=True)
# s = f"{path1}/seq{train_args['seq_len']}_32-576.log"
s = f"{path1}/profiler.log"
logger = get_logger(s)
logger.info(s)
train_configs = SimpleNamespace(**train_args)

map = {
    'GECCO': (9, '../dataset/GECCO/', 5),
    # 'MSL': (55, '../dataset/MSL/', 5),
    # 'SMAP': (25, '../dataset/SMAP/', 2),
    # 'PSM': (25, '../dataset/PSM/', 3),
    # 'Genesis': (18, '../dataset/Genesis/', 1),
    # 'NYC': (3, '../dataset/NYC/', 1),
    # 'Creditcard': (29, '../dataset/Creditcard/', 1),
    # 'SMD': (38, '../dataset/SMD/', 5),
    # 'CICIDS': (72, '../dataset/CICIDS/', 0.128),
}

with open(f'../configs/forecaster/{train_configs.forecaster}.json', 'r') as f:
    fst_args = json.load(f)


def split_train_val(data, val_ratio=0.2):
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def forecast_inference(fst, data, fcast_config, device):
    fst.eval()
    loader = provide_forecasting_data_loader(
        data, fcast_config.seq_len, fcast_config.label_len,
        fcast_config.pred_len, fcast_config.batch_size, shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train_configs.forecaster == 'autoformer':
                label_part = batch_y[:, :fcast_config.label_len, :]
                zeros_part = torch.zeros(
                    [batch_x.shape[0], fcast_config.pred_len, batch_x.shape[2]],
                    device=device
                )
                dec_inp = torch.cat([label_part, zeros_part], dim=1)
                outputs = fst(batch_x, None, dec_inp, None)
                f_dim = -1 if fcast_config.c_out == 1 else 0
                pred_out = outputs[:, -fcast_config.pred_len:, f_dim:]
            elif train_configs.forecaster == 'duet':
                pred_out, _ = fst(batch_x)
            elif train_configs.forecaster == 'patchtst':
                pred_out = fst(batch_x)[:, -fcast_config.pred_len:, :]
            elif train_configs.forecaster == 'dlinear':
                pred_out = fst(batch_x)
            elif train_configs.forecaster == 'timemixer':
                dec_inp = torch.zeros_like(batch_y[:, -fcast_config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :fcast_config.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = fst(batch_x, None, dec_inp, None)
                pred_out = outputs[:, -fcast_config.pred_len:, 0:]
            else:
                raise NotImplementedError

            predictions.append(pred_out.float().half().cpu().numpy())

    all_pred_windows = np.concatenate(predictions, axis=0).astype(np.float16)
    n_windows = all_pred_windows.shape[0]
    pred_length = n_windows + fcast_config.pred_len - 1

    reconstructed_preds = reconstruct_from_windows(
        all_windows=all_pred_windows,
        original_length=pred_length,
        seq_len=fcast_config.pred_len
    )
    return reconstructed_preds.astype(np.float32)


def train_test(train_data, test_data, fcast_config, anomaly_ratio, feature_dim):
    device = train_configs.device

    # ==================== 第一阶段：预测训练（forecast -> detect） ====================
    f_train_data, f_val_data = split_train_val(train_data, val_ratio=0.2)

    fst_train_loader = provide_forecasting_data_loader(
        f_train_data, fcast_config.seq_len, fcast_config.label_len,
        fcast_config.pred_len, fcast_config.batch_size, shuffle=True
    )

    fst_val_loader = provide_forecasting_data_loader(
        f_val_data, fcast_config.seq_len, fcast_config.label_len,
        fcast_config.pred_len, fcast_config.batch_size, shuffle=False
    )

    criterion = torch.nn.MSELoss(reduction='mean')
    if train_configs.forecaster == 'autoformer':
        fst = Autoformer(fcast_config).to(device)
    elif train_configs.forecaster == 'duet':
        fst = Duet(fcast_config).to(device)
    elif train_configs.forecaster == 'patchtst':
        fst = PatchTST(fcast_config).to(device)
    elif train_configs.forecaster == 'dlinear':
        fst = DLinear(fcast_config).to(device)
    elif train_configs.forecaster == 'timemixer':
        fst = TimeMixer(fcast_config).to(device)
    else:
        raise NotImplementedError

    optimizerF = torch.optim.Adam(list(fst.parameters()), lr=fcast_config.learning_rate)

    best_val_f = float('inf')
    best_model_f = None
    patience_counter_f = 0
    patience_f = 3
    min_delta_f = 1e-4

    for epoch in range(fcast_config.epochs):
        fst.train()
        total_loss = 0.0

        for batch_x, batch_y in fst_train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizerF.zero_grad()

            if train_configs.forecaster == 'autoformer':
                label_part = batch_y[:, :fcast_config.label_len, :]
                zeros_part = torch.zeros(
                    [batch_y.shape[0], fcast_config.pred_len, batch_y.shape[2]],
                    device=device
                )
                dec_inp = torch.cat([label_part, zeros_part], dim=1)
                outputs = fst(batch_x, None, dec_inp, None)
                f_dim = -1 if fcast_config.c_out == 1 else 0
                pred_out = outputs[:, -fcast_config.pred_len:, f_dim:]
                true_y = batch_y[:, -fcast_config.pred_len:, f_dim:]
                loss = criterion(pred_out, true_y)
            elif train_configs.forecaster == 'duet':
                pred_out, l_imp = fst(batch_x)
                loss = criterion(pred_out, batch_y) + l_imp
            elif train_configs.forecaster == 'patchtst':
                pred_out = fst(batch_x)[:, -fcast_config.pred_len:, :]
                loss = criterion(pred_out, batch_y)
            elif train_configs.forecaster == 'dlinear':
                pred_out = fst(batch_x)
                loss = criterion(pred_out, batch_y)
            elif train_configs.forecaster == 'timemixer':
                dec_inp = torch.zeros_like(batch_y[:, -fcast_config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :fcast_config.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = fst(batch_x, None, dec_inp, None)
                outputs = outputs[:, -fcast_config.pred_len:, 0:]
                batch_y = batch_y[:, -fcast_config.pred_len:, 0:]
                loss = criterion(outputs, batch_y)
            else:
                raise NotImplementedError

            loss.backward()
            optimizerF.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(fst_train_loader)

        fst.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in fst_val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if train_configs.forecaster == 'autoformer':
                    label_part = batch_y[:, :fcast_config.label_len, :]
                    zeros_part = torch.zeros(
                        [batch_y.shape[0], fcast_config.pred_len, batch_y.shape[2]],
                        device=device
                    )
                    dec_inp = torch.cat([label_part, zeros_part], dim=1)
                    outputs = fst(batch_x, None, dec_inp, None)
                    f_dim = -1 if fcast_config.c_out == 1 else 0
                    pred_out = outputs[:, -fcast_config.pred_len:, f_dim:]
                    true_y = batch_y[:, -fcast_config.pred_len:, f_dim:]
                    vloss = criterion(pred_out, true_y)
                elif train_configs.forecaster == 'duet':
                    pred_out, l_imp = fst(batch_x)
                    vloss = criterion(pred_out, batch_y) + l_imp
                elif train_configs.forecaster == 'patchtst':
                    pred_out = fst(batch_x)[:, -fcast_config.pred_len:, :]
                    vloss = criterion(pred_out, batch_y)
                elif train_configs.forecaster == 'dlinear':
                    pred_out = fst(batch_x)
                    vloss = criterion(pred_out, batch_y)
                elif train_configs.forecaster == 'timemixer':
                    dec_inp = torch.zeros_like(batch_y[:, -fcast_config.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :fcast_config.label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = fst(batch_x, None, dec_inp, None)
                    outputs = outputs[:, -fcast_config.pred_len:, 0:]
                    batch_y = batch_y[:, -fcast_config.pred_len:, 0:]
                    vloss = criterion(outputs, batch_y)
                else:
                    raise NotImplementedError

                val_loss_total += vloss.item()

        avg_val_loss = val_loss_total / len(fst_val_loader) if len(fst_val_loader) > 0 else avg_loss
        logger.info(
            f"Stage 1 (Forecast) - Epoch {epoch + 1}/{fcast_config.epochs}, "
            f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if (best_val_f - avg_val_loss) > min_delta_f:
            best_val_f = avg_val_loss
            best_model_f = fst.state_dict()
            patience_counter_f = 0
        else:
            patience_counter_f += 1

        if patience_counter_f >= patience_f:
            logger.info(f"Stage 1 (Forecast): Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_f is not None:
        fst.load_state_dict(best_model_f)

    # 预测得到 train/test 的 forecast 序列
    forecast_train = forecast_inference(fst, train_data, fcast_config, device)
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
    ) as prof:
        forecast_test = forecast_inference(fst, test_data, fcast_config, device)
    total_mflops = 0.0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_mflops += evt.flops / 1e6  # 转成 MFLOPs

    if forecast_train.ndim == 1:
        forecast_train = forecast_train[:, None]
    if forecast_test.ndim == 1:
        forecast_test = forecast_test[:, None]

    # ==================== 第二阶段：ModernTCN ====================
    col_names = [f'col_{i}' for i in range(feature_dim)]
    train_df = pd.DataFrame(forecast_train, columns=col_names)
    test_df = pd.DataFrame(forecast_test, columns=col_names)

    model = ModernTCN(
        seq_len=train_configs.seq_len,
        batch_size=256*2,
        lr=1e-4,
        num_epochs=1,
        enc_in=feature_dim,
        anomaly_ratio=anomaly_ratio
    )
    logger.info("Stage 2 (ModernTCN) - start training...")
    model.detect_fit(train_df, test_df)

    logger.info("Stage 2 (ModernTCN) - start inferencing...")
    train_score = model.detect_score(train_df)
    train_score = train_score.mean(axis=1)
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
    ) as prof2:
        test_score = model.detect_score(test_df)
    # total_mflops = 0.0
    for evt in prof2.key_averages():
        if evt.flops is not None:
            total_mflops += evt.flops / 1e6  # 转成 MFLOPs

    total_gflops = total_mflops / 1000.0
    logger.info(f"Test inference total FLOPs: {total_gflops:.4f} GFLOPs")
    total_params = count_params(model.model) + count_params(fst)
    logger.info(f'Total model parameters: {total_params:,}')
    exit()
    test_score = test_score.mean(axis=1)

    combined_energy = np.concatenate([train_score, test_score], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)

    return test_score, thresh


if __name__ == '__main__':
    pred_lens = [32]
    seeds = [3047]

    for p in pred_lens:
        train_configs.pred_len = p
        logger.info(f"Running experiments with pred_len={p}")
        logger.info(f'seq_len: {train_configs.seq_len}, pred_len: {train_configs.pred_len}')

        for dataset_name, (feature_dim, dataset_path, anomaly_ratio) in map.items():
            logger.info(f"Processing dataset: {dataset_name}")
            train_filepath = f"{dataset_path}train.csv"
            test_filepath = f"{dataset_path}test.csv"
            train_data = pd.read_csv(train_filepath)
            test_data = pd.read_csv(test_filepath)
            test_label = test_data['label']

            feature_cols = [f'col_{i}' for i in range(feature_dim)]
            train_medians = train_data[feature_cols].median()
            train_data[feature_cols] = train_data[feature_cols].fillna(train_medians)

            # load forecaster configs
            fst_args['enc_in'] = feature_dim
            fst_args['dec_in'] = feature_dim
            fst_args['c_out'] = feature_dim
            fst_args['seq_len'] = train_configs.seq_len
            fst_args['pred_len'] = train_configs.pred_len
            fst_args['device'] = train_configs.device
            fst_args['batch_size'] = 256
            fst_args['epochs'] = 1
            if dataset_name in ['SMAP']:
                fst_args['learning_rate'] = 1e-4
            fst_configs = SimpleNamespace(**fst_args)

            train_score = train_data[feature_cols].values
            test_score = test_data[feature_cols].values
            if train_configs.forecaster == 'duet':
                mean = train_score.mean(axis=0, keepdims=True)
                std = train_score.std(axis=0, keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)

                train_score = (train_score - mean) / std
                test_score = (test_score - mean) / std


            for seed in seeds:
                logger.info(f"\nRunning with seed: {seed}")
                set_seed(seed)
                scores, thresh = train_test(
                    train_score,
                    test_score,
                    fst_configs,
                    anomaly_ratio,
                    feature_dim
                )
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

                label = test_label.values
                labels = label[train_configs.seq_len:]
                metrics2(scores, labels[:len(scores)], thresh, logger)

            logger.info(f"{dataset_name} finished for pred_len={p}.\n-----------------------------------\n")
        logger.info(f"Finished all datasets for pred_len={p}\n============================\n")
