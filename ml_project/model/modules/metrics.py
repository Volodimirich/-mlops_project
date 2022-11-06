import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_auc_score

from entities import BadMetric


def metric_calc(x: pd.DataFrame, y: pd.DataFrame, metric: str = 'accuracy')\
        -> float:

    metric_dict = {'accuracy': accuracy_score,
                   'precision': precision_score,
                   'recall': recall_score,
                   'roc_auc': roc_auc_score}
    if metric not in metric_dict:
        raise BadMetric

    return metric_dict[metric](x, y)
