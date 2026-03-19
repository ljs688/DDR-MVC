from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_true.shape[0] == y_pred.shape[0]

    dim = int(max(y_true.max(), y_pred.max()) + 1)
    weight = np.zeros((dim, dim), dtype=np.int64)
    for idx in range(y_pred.shape[0]):
        weight[y_pred[idx], y_true[idx]] += 1

    row, col = linear_sum_assignment(weight.max() - weight)
    return float(weight[row, col].sum() / y_pred.shape[0])


def clustering_metrics(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> Tuple[float, float, float]:
    estimator = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred = estimator.fit_predict(features)

    acc = cluster_acc(labels, pred)
    nmi = float(normalized_mutual_info_score(labels, pred))
    ari = float(adjusted_rand_score(labels, pred))
    return acc, nmi, ari
