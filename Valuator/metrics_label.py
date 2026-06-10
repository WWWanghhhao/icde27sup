import numpy as np
from sklearn import metrics
from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .vus_metrics import metricor
from sklearn.metrics import accuracy_score




__all__ = [
    "accuracy",
    "f_score",
    "precision",
    "recall",
    "rrecall",
    "rprecision",
    "precision_at_k",
    "rf",
    "affiliation_f",
    "affiliation_precision",
    "affiliation_recall",
]

metricor_grader = metricor()


def precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]


def recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]


def f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]


def accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def rrecall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rrecall


def rprecision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rprecision


def rf(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return RF


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Precision_at_k


def affiliation_f(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return F


def affiliation_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P


def affiliation_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return R


def combined_1(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    accuracy = accuracy_score(actual, predicted)
    return Precision[1], Recall[1], F[1], accuracy

def combined_2(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rrecall, Rprecision, RF, Precision_at_k

def combined_3(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return F, P, R

def compute_label_metrics(actual, predicted):
    results = {}
    precision_value, recall_value, f_score_value, accuracy_value = combined_1(actual, predicted)
    results['precision'] = precision_value
    results['recall'] = recall_value
    results['f_score'] = f_score_value
    results['accuracy'] = accuracy_value

    rrecall_value, rprecision_value, rf_value, precision_at_k_value = combined_2(actual, predicted)
    results['rrecall'] = rrecall_value
    results['rprecision'] = rprecision_value
    results['rf'] = rf_value
    results['precision_at_k'] = precision_at_k_value

    F, P, R = combined_3(actual, predicted)
    results['affiliation_f'] = F
    results['affiliation_precision'] = P
    results['affiliation_recall'] = R

    # for metric_name in __all__:
    #     metric_func = globals()[metric_name]
    #     results[metric_name] = metric_func(actual, predicted)
    return results


# score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
#
#
# print("precision:", precision(label, score,))
# print("recall:", recall(label, score))
# print("f_score:", f_score(label, score,))
# print("rrecall:", rrecall(label, score,))
# print("rprecision:", rprecision(label, score,))
# print("rf:", rf(label, score,))
# print("precision_at_k:", precision_at_k(label, score,))
