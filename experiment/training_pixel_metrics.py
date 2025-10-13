# -*- coding: utf-8 -*-

"""
    The training pixel metrics module
    ======================

    Use it to compute different metrics during training.
    Available metrics:
        - Confusion matrix
        - Intersection-over-Union
"""

import numpy as np


def compute_metrics(
    pred: np.ndarray, label: np.ndarray, loss: float, classes: list
) -> dict:

    metrics = {}
    metrics["matrix"] = confusion_matrix(pred, label, classes)
    metrics["loss"] = loss
    return metrics


def update_metrics(metrics: dict, batch_metrics: dict,index:int) -> dict:

    for i in range(metrics["matrix"].shape[0]):
        for j in range(metrics["matrix"].shape[1]):
            metrics["matrix"][i][j] += batch_metrics["matrix"][i][j]
    #  保证一个batch加一次
    if index==0:
        metrics["loss"] += batch_metrics["loss"]
    return metrics


def confusion_matrix(pred, label, classes: list) -> np.array:

    size = len(classes)
    confusion_matrix = np.zeros((size, size))

    confusion_matrix[label, pred] = 1
    return confusion_matrix


def iou(confusion_matrix: np.ndarray, channel: str) -> float:

    TP = confusion_matrix[channel, channel]
    # 漏检+TP
    tpfn = np.sum(confusion_matrix[channel,:])
    FN=tpfn-TP
    # 误检+TP
    tpfp = np.sum(confusion_matrix[:, channel])
    FP=tpfp-TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)

    if TP==0 or TP+FN==0:
        recall=0
    else:
        recall = TP / (TP + FN)
    if TP==0 or TP+FP==0:
        pre=0
    else:
        pre = TP / (TP + FP)
    if TP+TN==0 or TP + TN + FN + FP==0:
        acc=0
    else:
        acc = (TP + TN) / (TP + TN + FN + FP)
    #     F1=2×(Precision*Recall)/(Precision+Recall)

    if TP==0 or 2 * TP + FP + FN==0:
        f1=0
    else:
        f1 = 2 * TP / (2 * TP + FP + FN)
    return recall,pre,f1
