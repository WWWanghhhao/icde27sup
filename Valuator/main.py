from .metrics_label import compute_label_metrics
from .metrics_score import compute_score_metrics
import numpy as np
import pandas as pd


def score_to_label(score_series, anomaly_ratio: float) -> pd.Series:
    percentile = 100 - (anomaly_ratio * 100)
    threshold = np.percentile(score_series, percentile)
    label_series = (score_series > threshold).astype(int)
    return label_series


def metrics2(scores, label, thresh, logger):
    logger.info("score metrics:")
    score_metrics = compute_score_metrics(label, scores)
    logger.info(f"" + ', '.join([f"{name}: {value:.4f}" for name, value in score_metrics.items()]))
    logger.info("label metrics:")
    pred_label = (scores > thresh).astype(int)
    label_metrics = compute_label_metrics(label, pred_label)
    logger.info(f"" + ', '.join([f"{name}: {value:.4f}" for name, value in label_metrics.items()]))




def metrics(scores, label, ratio, logger):
    # logger.info("Starting metrics phase...")
    score_metrics = compute_score_metrics(label, scores)

    logger.info(f"" + ', '.join([f"{name}: {value:.4f}" for name, value in score_metrics.items()]))

    ls = [round(x * 0.02, 2) for x in range(1, 10)]
    best = {}  # metric_name -> (best_value, best_ratio)
    for ratio in ls:
        pred_label = score_to_label(scores, ratio)
        label_metrics = compute_label_metrics(label, pred_label)
        for metric_name, metric_value in label_metrics.items():
            if metric_name not in best or metric_value > best[metric_name][0]:
                best[metric_name] = (metric_value, ratio)

    # 只输出每个指标的最优值和对应的 ratio
    for metric_name, (best_value, best_ratio) in best.items():
        logger.info(f"Best {metric_name}: {best_value:.4f} at ratio: {best_ratio:.2f}")
