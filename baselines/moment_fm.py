import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from momentfm import MOMENTPipeline

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

from Valuator.main import metrics2
from Utils.tools import get_logger, provide_data_loader, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MOMENT forecasting-head training and anomaly prediction")

    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=192, help="history length used by the anomaly prediction task")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[32], help="forecast horizons to evaluate")
    parser.add_argument("--moment_context_len", type=int, default=512, help="MOMENT input context length")
    parser.add_argument("--datasets", type=str, nargs="+", default=["SMD"], help="dataset names in map keys")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3047])

    parser.add_argument("--epochs", type=int, default=1, help="epochs for the forecasting head")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="moment_weights/MOMENT-1-small")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder")

    parser.add_argument("--score_reduce", type=str, default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--log_dir", type=str, default="./logs/moment/forecast_anomaly")

    return parser.parse_args()

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_model_params(logger, name, model):
    total, trainable = count_params(model)
    logger.info(f"Params - {name}: total={total:,}, trainable={trainable:,}")
    return total, trainable

def split_train_val(data, val_ratio=0.2):
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def ensure_context(x, input_mask, context_len):
    """Pad or crop [B, C, T] input to MOMENT's context length."""
    if x.size(-1) == context_len:
        return x, input_mask

    if x.size(-1) > context_len:
        return x[..., -context_len:], input_mask[..., -context_len:]

    pad_len = context_len - x.size(-1)
    x = torch.nn.functional.pad(x, (0, pad_len))
    input_mask = torch.nn.functional.pad(input_mask, (0, pad_len))
    return x, input_mask


def unpack_window_batch(batch, seq_len, pred_len, device):
    batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
    batch_data = batch_data.to(device).float()

    history = batch_data[:, :seq_len, :].transpose(1, 2).contiguous()
    future = batch_data[:, seq_len:seq_len + pred_len, :].transpose(1, 2).contiguous()
    input_mask = torch.ones(history.size(0), history.size(-1), device=device, dtype=torch.long)
    return history, future, input_mask


def build_forecasting_model(args, n_channels, pred_len):
    model = MOMENTPipeline.from_pretrained(
        args.model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": pred_len,
            "head_dropout": args.head_dropout,
            "weight_decay": args.weight_decay,
            "freeze_encoder": args.freeze_encoder,
            "freeze_embedder": args.freeze_encoder,
            "freeze_head": False,
        },
        local_files_only=args.local_files_only,
    )
    model.init()

    for name, param in model.named_parameters():
        param.requires_grad = "head" in name

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_names:
        for name, param in model.named_parameters():
            if any(token in name.lower() for token in ("forecast", "linear", "projection")):
                param.requires_grad = True

    return model


def build_reconstruction_model(args):
    model = MOMENTPipeline.from_pretrained(
        args.model_name,
        model_kwargs={"task_name": "reconstruction"},
        local_files_only=args.local_files_only,
    )
    model.init()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def train_forecasting_head(model, train_loader, val_loader, args, pred_len, logger):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model.to(device).float()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check MOMENT head parameter names.")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in tqdm(train_loader, total=len(train_loader), desc=f"Forecast epoch {epoch + 1}/{args.epochs}"):
            history, future, input_mask = unpack_window_batch(batch, args.seq_len, pred_len, device)
            history, input_mask = ensure_context(history, input_mask, args.moment_context_len)

            optimizer.zero_grad(set_to_none=True)
            output = model(x_enc=history, input_mask=input_mask)
            loss = criterion(output.forecast, future)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss = evaluate_forecast_loss(model, val_loader, args, pred_len, criterion, device)
        train_loss = float(np.mean(losses)) if losses else 0.0
        logger.info(f"Forecast head epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_forecast_loss(model, data_loader, args, pred_len, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            history, future, input_mask = unpack_window_batch(batch, args.seq_len, pred_len, device)
            history, input_mask = ensure_context(history, input_mask, args.moment_context_len)
            output = model(x_enc=history, input_mask=input_mask)
            losses.append(criterion(output.forecast, future).item())
    return float(np.mean(losses)) if losses else 0.0


def zero_shot_scores_on_forecasts(forecaster, reconstructor, data_loader, args, pred_len):
    device = next(forecaster.parameters()).device
    forecaster.eval()
    reconstructor.to(device).float().eval()

    scores = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Zero-shot anomaly scoring"):
            history, _, input_mask = unpack_window_batch(batch, args.seq_len, pred_len, device)
            history, input_mask = ensure_context(history, input_mask, args.moment_context_len)

            forecast = forecaster(x_enc=history, input_mask=input_mask).forecast
            forecast_mask = torch.ones(forecast.size(0), forecast.size(-1), device=device, dtype=torch.long)
            forecast_input, forecast_mask = ensure_context(forecast, forecast_mask, args.moment_context_len)

            reconstruction = reconstructor(x_enc=forecast_input, input_mask=forecast_mask).reconstruction
            reconstruction = reconstruction[..., :pred_len]

            step_score = torch.mean((forecast - reconstruction) ** 2, dim=1)
            if args.score_reduce == "max":
                batch_score = step_score.max(dim=1).values
            elif args.score_reduce == "last":
                batch_score = step_score[:, -1]
            else:
                batch_score = step_score.mean(dim=1)
            scores.append(batch_score.detach().cpu().numpy())

    return np.concatenate(scores) if scores else np.array([])


def train_test(train_data, test_data, dataset_name, anomaly_ratio, args, pred_len, logger):
    train_df, val_df = split_train_val(train_data, val_ratio=args.val_ratio)

    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_df.values)
    scaled_val = scaler.transform(val_df.values)
    scaled_test = scaler.transform(test_data.values)

    window_len = args.seq_len + pred_len
    train_loader = provide_data_loader(
        scaled_train,
        window_len,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = provide_data_loader(
        scaled_val,
        window_len,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = provide_data_loader(
        scaled_test,
        window_len,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    n_channels = train_data.shape[1]
    logger.info(f"Loading MOMENT forecasting model: channels={n_channels}, seq_len={args.seq_len}, pred_len={pred_len}")
    forecaster = build_forecasting_model(args, n_channels=n_channels, pred_len=pred_len)
    f_total, f_trainable = log_model_params(logger, "MOMENT forecasting", forecaster)
    forecaster = train_forecasting_head(forecaster, train_loader, val_loader, args, pred_len, logger)

    logger.info("Loading MOMENT reconstruction model for zero-shot anomaly scoring on forecasts")
    reconstructor = build_reconstruction_model(args)
    r_total, r_trainable = log_model_params(logger, "MOMENT reconstruction", reconstructor)
    logger.info(
        f"Params - zero-shot pipeline: total={f_total + r_total:,}, "
        f"trainable={f_trainable + r_trainable:,}"
    )

    train_scores = zero_shot_scores_on_forecasts(forecaster, reconstructor, train_loader, args, pred_len)
    test_scores = zero_shot_scores_on_forecasts(forecaster, reconstructor, test_loader, args, pred_len)

    thresh = np.percentile(np.concatenate([train_scores, test_scores]), 100 - anomaly_ratio)
    logger.info(
        f"{dataset_name}: score_min={test_scores.min():.6f}, score_max={test_scores.max():.6f}, "
        f"score_std={test_scores.std():.6f}, thresh={thresh:.6f}"
    )
    return test_scores, thresh


def load_feature_frame(train_data, test_data, feature_dim):
    feature_cols = [f"col_{i}" for i in range(feature_dim)]
    train_medians = train_data[feature_cols].median()
    train_data[feature_cols] = train_data[feature_cols].fillna(train_medians)
    test_data[feature_cols] = test_data[feature_cols].fillna(train_medians)
    return train_data[feature_cols], test_data[feature_cols]


def build_window_labels(labels, seq_len, pred_len, n_windows):
    window_labels = []
    for i in range(n_windows):
        start = i + seq_len
        end = start + pred_len
        if end > len(labels):
            break
        window_labels.append(int(np.any(labels[start:end] > 0)))
    return np.asarray(window_labels, dtype=int)


args = parse_args()

os.makedirs(args.log_dir, exist_ok=True)
log_path = os.path.join(args.log_dir, f"zero{args.datasets}_{args.pred_lens}.log")
logger = get_logger(log_path)
logger.info(log_path)
logger.info("=" * 50)
logger.info("Command-line arguments:")
for arg_name, arg_value in vars(args).items():
    logger.info(f"  {arg_name}: {arg_value}")

dataset_map = {
    "GECCO": (9, "../dataset/GECCO/", 1),
    "MSL": (55, "../dataset/MSL/", 5),
    "SMAP": (25, "../dataset/SMAP/", 2),
    "PSM": (25, "../dataset/PSM/", 5),
    "Genesis": (18, "../dataset/Genesis/", 5),
    "NYC": (3, "../dataset/NYC/", 5),
    "Creditcard": (29, "../dataset/Creditcard/", 5),
    "SMD": (38, "../dataset/SMD/", 5),
}

try:
    with open("../configs/config.json", "r") as f:
        config_map = json.load(f)
except FileNotFoundError:
    config_map = {}

train_configs = SimpleNamespace(seq_len=args.seq_len, device=args.device)


if __name__ == "__main__":
    selected_datasets = {k: v for k, v in dataset_map.items() if k in args.datasets}

    for pred_len in args.pred_lens:
        train_configs.pred_len = pred_len

        for dataset_name, (feature_dim, dataset_path, anomaly_ratio) in selected_datasets.items():
            logger.info(f"Processing dataset: {dataset_name}, seq_len={args.seq_len}, pred_len={pred_len}")

            train_filepath = os.path.join(dataset_path, "train.csv")
            test_filepath = os.path.join(dataset_path, "test.csv")

            train_data = pd.read_csv(train_filepath)
            test_data = pd.read_csv(test_filepath)
            test_label = test_data["label"].values

            train_features, test_features = load_feature_frame(train_data, test_data, feature_dim)
            if dataset_name in config_map and "anomaly_ratio" in config_map[dataset_name]:
                anomaly_ratio = config_map[dataset_name]["anomaly_ratio"]

            for seed in args.seeds:
                logger.info(f"\nRunning with seed: {seed}")
                set_seed(seed)

                scores, thresh = train_test(
                    train_features,
                    test_features,
                    dataset_name=dataset_name,
                    anomaly_ratio=anomaly_ratio,
                    args=args,
                    pred_len=pred_len,
                    logger=logger,
                )

                labels = build_window_labels(test_label, args.seq_len, pred_len, len(scores))
                scores = scores[:len(labels)]
                metrics2(scores, labels, thresh, logger)

            logger.info(f"{dataset_name} finished.\n-----------------------------------\n")
