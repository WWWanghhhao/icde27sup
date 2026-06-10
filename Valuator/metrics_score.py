import numpy as np
from sklearn import metrics
from .vus_metrics import metricor, generate_curve
from sklearn.metrics import precision_recall_curve
from typing import List, Dict, Iterable, Any


def get_list_anomaly(labels: np.ndarray) -> List[int]:
    """
    获取时间序列标签中的异常间隔长度列表。

    :param labels: 时间序列标签列表，1 表示异常，0 表示正常。
    :return: 异常间隔长度列表。
    """
    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)


# __all__ = ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]
__all__ = ["best_ratio", "best_accuracy", "best_f_score", "best_precision", "best_recall", "auc_roc", "auc_pr",
           "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]

metricor_grader = metricor()


def best_ratio(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return anomaly_rate


def best_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return best_f1_score


def best_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return accuracy


def best_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return recall


def best_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)

    return precision


def auc_roc(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.roc_auc_score(actual, predicted)


def auc_pr(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.average_precision_score(actual, predicted)


def R_AUC_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_ROC


def R_AUC_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_PR


def VUS_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_ROC


def VUS_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_PR


def combined_1(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return best_f1_score, accuracy, precision, recall, anomaly_rate


def combined_2(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    auc_roc_value = metrics.roc_auc_score(actual, predicted)
    auc_pr_value = metrics.average_precision_score(actual, predicted)
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    R_AUC_ROC_value, R_AUC_PR_value, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    _, _, _, _, _, _, VUS_ROC_value, VUS_PR_value = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return auc_roc_value, auc_pr_value, R_AUC_ROC_value, R_AUC_PR_value, VUS_ROC_value, VUS_PR_value


def compute_score_metrics(actual, predicted):
    results = {}
    best_f_score, accuracy, precision, recall, anomaly_rate = combined_1(actual, predicted)
    results["best_f_score"] = best_f_score
    results["best_accuracy"] = accuracy
    results["best_precision"] = precision
    results["best_recall"] = recall
    results["best_ratio"] = anomaly_rate

    auc_roc_value, auc_pr_value, R_AUC_ROC_value, R_AUC_PR_value, VUS_ROC_value, VUS_PR_value = combined_2(actual, predicted)
    results["auc_roc"] = auc_roc_value
    results["auc_pr"] = auc_pr_value
    results["R_AUC_ROC"] = R_AUC_ROC_value
    results["R_AUC_PR"] = R_AUC_PR_value
    results["VUS_ROC"] = VUS_ROC_value
    results["VUS_PR"] = VUS_PR_value


    # for metric_name in __all__:
    #     metric_func = globals()[metric_name]
    #     results[metric_name] = metric_func(actual, predicted)
    return results



# y_test = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
# pred_labels = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
#
# res = compute_score_metrics(y_test, pred_labels)
# print(res)



# a = VUS_ROC(y_test, pred_labels)
# b = VUS_PR(y_test, pred_labels)
# # vus_results = get_range_vus_roc(y_test, pred_labels, 100)  # default slidingWindow = 100
# print("VUS_ROC", a)
# print("VUS_PR", b)
#
#
# import numpy as np
#
#
# score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
# print("auc_roc:", auc_roc(label, score))
# print("auc_pr:", auc_pr(label, score))
# print("R_AUC_ROC:", R_AUC_ROC(label, score))
# print("R_AUC_PR:", R_AUC_PR(label, score))
#
# print("VUS_ROC:", VUS_ROC(label, score))
# print("VUS_PR:", VUS_PR(label, score))
