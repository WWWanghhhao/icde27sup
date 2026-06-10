import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
import sys, os, json, math
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

from Valuator.main import metrics2
from Utils.tools import provide_data_loader, get_logger, set_seed, provide_forecasting_data_loader, \
    reconstruct_from_windows, my_kl_loss
from Detector.CATCH_layers.freq import frequency_loss
from Detector.CATCH import CATCHModel
from Detector.SELSTM import SELSTMModel as Selstm
from Detector.AnomalyTransformer import AnomalyTransformer
from Detector.DCdetector import DCdetector
from Detector.KANAD import KANAD
from Detector.FusAD import Model as FusAD
from Forecaster.Autoformer import Model as Autoformer
from Forecaster.DUET import DUET as Duet
from Forecaster.PatchTST import Model as PatchTST
from Forecaster.DLinear import Model as DLinear
from Forecaster.TimeMixer import Model as TimeMixer

from Processor.inference import model_inference

train_args = {
    "seq_len": 192,
    "pred_len": 128,
    "device": "cuda:0",
    'detector': 'kanad',
    'forecaster': 'patchtst',
}

path1 = f"./logs/f_d/thresh/{train_args['forecaster']}-{train_args['detector']}"
if not os.path.exists(path1):
    os.makedirs(path1)
# s = f"{path1}/seq{train_args['seq_len']}_32-576.log"
s = f"{path1}/profiler.log"
print(s)
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

    # 'SWaT': (51, './dataset/SWaT/', 0.121),
}

with open(f'../configs/detector/{train_configs.detector}.json', 'r') as f:
    det_args = json.load(f)
with open(f'../configs/forecaster/{train_configs.forecaster}.json', 'r') as f:
    fst_args = json.load(f)

def count_params(model):
    return sum(p.numel() for p in model.parameters())
def split_train_val(data, val_ratio=0.2):
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


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
    # summed_preds = reconstructed_preds.mean(axis=1)
    return reconstructed_preds.astype(np.float32)


def train_test(train_data, test_data, detect_config, fcast_config, anomaly_ratio):
    device = train_configs.device
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data.values)
    scaled_test = scaler.transform(test_data.values)

    # ==================== 第一阶段：预测训练（forecast -> detect） ====================
    f_train_data, f_val_data = split_train_val(scaled_train, val_ratio=0.2)

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

    # forecast 早停机制（基于验证集 loss 下降幅度）
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

        # 验证集 loss
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
        if len(fst_val_loader) > 0:
            avg_val_loss = val_loss_total / len(fst_val_loader)
        else:
            avg_val_loss = avg_loss

        logger.info(
            f"Stage 1 (Forecast) - Epoch {epoch + 1}/{fcast_config.epochs}, "
            f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # 早停判断：验证集 loss 下降幅度小于阈值
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
    forecast_train = forecast_inference(fst, scaled_train, fcast_config, device)
    # print(forecast_train.shape)
    # exit()
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
    ) as prof2:
        forecast_test = forecast_inference(fst, scaled_test, fcast_config, device)
    total_mflops = 0.0
    for evt in prof2.key_averages():
        if evt.flops is not None:
            total_mflops += evt.flops / 1e6  # 转成 MFLOPs
    total_gflops = total_mflops / 1000.0
    logger.info(f"Test inference1 total FLOPs: {total_gflops:.4f} GFLOPs")



    if forecast_train.ndim == 1:
        forecast_train = forecast_train[:, None]
    if forecast_test.ndim == 1:
        forecast_test = forecast_test[:, None]

    # ==================== 第二阶段：异常检测训练 ====================
    d_train_data, d_val_data = split_train_val(forecast_train, val_ratio=0.2)

    det_train_loader = provide_data_loader(d_train_data, detect_config.seq_len, detect_config.batch_size,
                                           shuffle=True, num_workers=0)
    det_val_loader = provide_data_loader(d_val_data, detect_config.seq_len, detect_config.batch_size,
                                         shuffle=False, num_workers=0)
    det_test_loader = provide_data_loader(forecast_test, detect_config.seq_len, detect_config.batch_size,
                                          shuffle=False, num_workers=0)

    if train_configs.detector == 'selstm':
        det = Selstm(detect_config).to(device)
    elif train_configs.detector == 'catch':
        det = CATCHModel(detect_config).to(device)
    elif train_configs.detector == 'anomalytransformer':
        det = AnomalyTransformer(detect_config).to(device)
    elif train_configs.detector == 'dcdetector':
        det = DCdetector(detect_config).to(device)
    elif train_configs.detector == 'kanad':
        det = KANAD(detect_config).to(device)
    elif train_configs.detector == 'fusad':
        det = FusAD(detect_config).to(device)
    else:
        raise NotImplementedError




    if train_configs.detector == 'catch':
        main_params = [param for name, param in det.named_parameters() if 'mask_generator' not in name]
        optimizerD = torch.optim.Adam(main_params, lr=detect_config.lr)
        optimizerM = torch.optim.Adam(det.mask_generator.parameters(), lr=detect_config.Mlr)
        auxi_loss = frequency_loss(detect_config)
    else:
        optimizerD = torch.optim.Adam(list(det.parameters()), lr=detect_config.learning_rate)
        optimizerM = None
        auxi_loss = None

    criterion = torch.nn.MSELoss(reduction='mean')

    # detect 早停机制（基于验证集 loss 下降幅度）
    best_val_d = float('inf')
    best_model_d = None
    patience_counter_d = 0
    patience_d = 3
    min_delta_d = 1e-6

    for epoch in range(detect_config.epochs):
        det.train()
        total_loss = 0.0

        for batch_data, _ in det_train_loader:
            batch_data = batch_data.to(device)
            optimizerD.zero_grad()

            if train_configs.detector == 'catch':
                optimizerM.zero_grad()
                rec_batch_data, freq_feats, dcloss = det(batch_data)
                fft_y = torch.fft.fft(batch_data)
                l_freq = auxi_loss(freq_feats, fft_y)
                loss = ((detect_config.dc_lambda * dcloss) + (detect_config.auxi_lambda * l_freq)
                        + criterion(batch_data, rec_batch_data))
            elif train_configs.detector == 'anomalytransformer':
                output, series, prior, _ = det(batch_data)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   train_configs.seq_len)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           train_configs.seq_len)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                train_configs.seq_len)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       train_configs.seq_len)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = criterion(output, batch_data)
                loss1 = rec_loss - detect_config.k * series_loss
                loss2 = rec_loss + detect_config.k * prior_loss

                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizerD.step()
                total_loss += loss1.item()
                continue
            elif train_configs.detector == 'dcdetector':
                series, prior = det(batch_data)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   train_configs.seq_len)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           train_configs.seq_len)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                train_configs.seq_len)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       train_configs.seq_len)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss
            elif train_configs.detector == 'fusad':
                rec_batch_data = det(batch_data, None, None, None)
                loss = criterion(batch_data, rec_batch_data)
            else:
                rec_batch_data = det(batch_data)
                loss = criterion(batch_data, rec_batch_data)

            loss.backward()
            optimizerD.step()
            if train_configs.detector == 'catch':
                optimizerM.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(det_train_loader)

        # 验证集 loss
        det.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_data, _ in det_val_loader:
                batch_data = batch_data.to(device)
                if train_configs.detector == 'catch':
                    rec_batch_data, freq_feats, dcloss = det(batch_data)
                    fft_y = torch.fft.fft(batch_data)
                    l_freq = auxi_loss(freq_feats, fft_y)
                    vloss = ((detect_config.dc_lambda * dcloss) + (detect_config.auxi_lambda * l_freq)
                             + criterion(batch_data, rec_batch_data))
                elif train_configs.detector == 'anomalytransformer':
                    output, series, prior, _ = det(batch_data)
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       train_configs.seq_len)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach(),
                                       series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    train_configs.seq_len)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           train_configs.seq_len)))))
                    series_loss = series_loss / len(prior)
                    rec_loss = criterion(output, batch_data)
                    vloss = rec_loss - detect_config.k * series_loss
                elif train_configs.detector == 'dcdetector':
                    series, prior = det(batch_data)
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       train_configs.seq_len)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach(),
                                       series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    train_configs.seq_len)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           train_configs.seq_len)))))
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)
                    vloss = prior_loss - series_loss
                else:
                    rec_batch_data = det(batch_data)
                    vloss = criterion(batch_data, rec_batch_data)

                val_loss_total += vloss.item()

        if len(det_val_loader) > 0:
            avg_val_loss = val_loss_total / len(det_val_loader)
        else:
            avg_val_loss = avg_loss
        logger.info(
            f"Stage 2 (Detect) - Epoch {epoch + 1}/{detect_config.epochs}, "
            f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # 早停判断：验证集 loss 下降幅度小于阈值
        if (best_val_d - avg_val_loss) > min_delta_d:
            best_val_d = avg_val_loss
            best_model_d = det.state_dict()
            patience_counter_d = 0
        else:
            patience_counter_d += 1

        if patience_counter_d >= patience_d:
            logger.info(f"Stage 2 (Detect): Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_d is not None:
        det.load_state_dict(best_model_d)

    # ==================== 测试阶段：输出最终 anomaly scores ====================
    def detect_scores(model, data_loader, original_length):
        model.eval()
        window_scores = []

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                for batch_data, _ in data_loader:
                    batch_data = batch_data.to(device)
                    if train_configs.detector == 'anomalytransformer':
                        temperature = 50
                        output, series, prior, _ = model(batch_data)
                        loss = torch.mean(criterion(batch_data, output), dim=-1)
                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            if u == 0:
                                series_loss = my_kl_loss(series[u], (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach()) * temperature
                                prior_loss = my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            train_configs.seq_len)),
                                    series[u].detach()) * temperature
                            else:
                                series_loss += my_kl_loss(series[u], (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach()) * temperature
                                prior_loss += my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            train_configs.seq_len)),
                                    series[u].detach()) * temperature

                        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                        scores = metric * loss
                    elif train_configs.detector == 'catch':
                        rec_batch_data, _, _ = model(batch_data)
                        scores = torch.mean((batch_data - rec_batch_data) ** 2, dim=-1)
                    elif train_configs.detector == 'dcdetector':
                        temperature = 50
                        series, prior = model(batch_data)
                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            if u == 0:
                                series_loss = my_kl_loss(series[u], (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach()) * temperature
                                prior_loss = my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            train_configs.seq_len)),
                                    series[u].detach()) * temperature
                            else:
                                series_loss += my_kl_loss(series[u], (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               train_configs.seq_len)).detach()) * temperature
                                prior_loss += my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            train_configs.seq_len)),
                                    series[u].detach()) * temperature

                        scores = torch.softmax((-series_loss - prior_loss), dim=-1)

                    else:
                        rec_batch_data = model(batch_data)
                        score = criterion(rec_batch_data, batch_data).detach()
                        scores = torch.mean(score, dim=-1)

                    # window_scores.append(scores.cpu().half().numpy())
        # 统计推理总 FLOPs
        total_mflops = 0.0
        for evt in prof.key_averages():
            if evt.flops is not None:
                total_mflops += evt.flops / 1e6  # 转成 MFLOPs

        total_gflops = total_mflops / 1000.0
        logger.info(f"Test inference total FLOPs: {total_gflops:.4f} GFLOPs")
        total_params = count_params(det) + count_params(fst)
        logger.info(f'Total model parameters: {total_params:,}')
        exit()

        all_windows = np.concatenate(window_scores, axis=0)
        reconstructed_scores = reconstruct_from_windows(
            all_windows=all_windows,
            original_length=original_length,
            seq_len=detect_config.seq_len
        )
        return reconstructed_scores.astype(np.float32)

    attens_energy = detect_scores(det, det_test_loader, original_length=len(forecast_test)).reshape(-1)
    test_energy = np.array(attens_energy)

    attens_energy = detect_scores(det, det_train_loader, original_length=len(forecast_train)).reshape(-1)
    train_energy = np.array(attens_energy)


    # min_val = train_energy.min()
    # max_val = train_energy.max()
    # train_energy = (train_energy - min_val) / (max_val - min_val)
    # test_energy = (test_energy - min_val) / (max_val - min_val)

    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
    return attens_energy, thresh


# python
if __name__ == '__main__':
    pred_lens = [32]  # 32, 64, 96, 128, 192, 384, 576
    seeds = [3047]  # 5948,

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

            # load configs
            if train_args['detector'] == 'catch':
                with open(f'../configs/detector/catch_presettings.json', 'r') as f:
                    settings = json.load(f)
                det_args.update(settings[dataset_name])
            det_args['enc_in'] = feature_dim
            det_args['c_out'] = feature_dim
            det_args['seq_len'] = train_configs.seq_len
            det_args['device'] = train_configs.device
            det_args['epochs'] = 1
            det_args['batch_size'] = 512
            # det_args['learning_rate'] = 1e-3
            det_configs = SimpleNamespace(**det_args)

            fst_args['enc_in'] = feature_dim
            fst_args['dec_in'] = feature_dim
            fst_args['c_out'] = feature_dim
            fst_args['seq_len'] = train_configs.seq_len
            fst_args['pred_len'] = train_configs.pred_len
            fst_args['device'] = train_configs.device
            fst_args['epochs'] = 1
            fst_args['batch_size'] = 512
            # fst_args['learning_rate'] = 1e-3
            if dataset_name in ['SMAP']:
                fst_args['learning_rate'] = 1e-4
            fst_configs = SimpleNamespace(**fst_args)

            for seed in seeds:
                logger.info(f"\nRunning with seed: {seed}")
                set_seed(seed)
                scores, thresh = train_test(train_data[feature_cols], test_data[feature_cols], det_configs, fst_configs,
                                            anomaly_ratio)
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

                label = test_label.values
                labels = label[train_configs.seq_len:]
                metrics2(scores, labels[:len(scores)], thresh, logger)

            logger.info(f"{dataset_name} finished for pred_len={p}.\n-----------------------------------\n")
        logger.info(f"Finished all datasets for pred_len={p}\n============================\n")
