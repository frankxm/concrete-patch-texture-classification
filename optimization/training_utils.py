# -*- coding: utf-8 -*-

"""
    The training utils module
    ======================

    Use it to during the training stage.
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import training_pixel_metrics as p_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelLoss(nn.Module):
    def __init__(self, num_classes, weights=None, alpha=0.6, reduction="mean",label_smooth=0.2):

        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        self.label_smooth = label_smooth

        if weights is not None:
            self.register_buffer(
                "weights",
                torch.as_tensor(weights, dtype=torch.float32)
            )
        else:
            self.weights = None
#label smooth+ soft label
    def build_soft_target(self, label1, label2):
        B = label1.size(0)
        device = label1.device

        target = torch.zeros(B, self.num_classes, device=device)

        for i in range(B):
            l1 = label1[i].item()
            target[i, l1] = self.alpha
            l2 = int(label2[i].item())
            if l2 >= 0:
                target[i, l2] += (1 - self.alpha)
        soft_target = (1 - self.label_smooth) * target + self.label_smooth / self.num_classes
        return soft_target
#label smooth
    def build_soft_target_ls(self, label1):
        B = label1.size(0)
        device = label1.device

        target = torch.zeros(B, self.num_classes, device=device)
        target[torch.arange(B), label1] = 1.0

        target = (1 - self.label_smooth) * target + self.label_smooth / self.num_classes

        return target

    def build_one_hot(self, label1):
        B = label1.size(0)
        device = label1.device
        target = torch.zeros(B, self.num_classes, device=device)
        target[torch.arange(B), label1] = 1.0
        return target

    def build_bootstrap_soft_target(self, output, label1, label2, beta):

        with torch.no_grad():
            # target = self.build_soft_target(label1, label2)
            target =self.build_one_hot(label1)
            pred_prob = F.softmax(output, dim=1)

            target_bootstrap = (
                    beta * target
                    + (1 - beta) * pred_prob
            )

        return target_bootstrap

    def build_bootstrap_hard_target(self, output, label1, label2, beta):

        with torch.no_grad():
            # target = self.build_soft_target2(label1, label2)
            target =self.build_one_hot(label1)
            pred_prob = F.softmax(output, dim=1)

            pseudo = torch.zeros_like(pred_prob)
            # 1表示dim=1沿着列方向填1.0
            pseudo.scatter_( 1,pred_prob.argmax(dim=1, keepdim=True), 1.0 )

            target_bootstrap = (
                    beta * target
                    + (1 - beta) * pseudo
            )

        return target_bootstrap
    def forward(self, output, label1, label2):

        log_probs = F.log_softmax(output, dim=1)
        # target = self.build_soft_target(label1, label2)
        # target = self.build_bootstrap_soft_target(output,label1,label2,beta=0.95)
        # target = self.build_bootstrap_hard_target(output,label1,label2,beta=0.8)
        target=self.build_soft_target_ls(label1)
        loss = -(target * log_probs).sum(dim=1)

        # class weighting
        if self.weights is not None:
            loss = loss * self.weights[label1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def get_epoch_values(metrics: dict, classes: list, num_samples_until_this_batch: int) -> dict:

    values = {}
    total_tp=0
    total_samples=0
    f1_scores = []
    for channel in classes:
        recall, precision, f1 = p_metrics.iou(metrics["matrix"], classes.index(channel))
        values["recall_" + channel] = round(recall, 4)
        values["precision_" + channel] = round(precision, 4)
        values["f1_" + channel] = round(f1, 4)

        # 累计总的TP和所有样本数
        total_tp += metrics["matrix"][classes.index(channel), classes.index(channel)]
        total_samples += np.sum(metrics["matrix"][classes.index(channel), :])
        f1_scores.append(f1)
        # 计算所有类的总准确率
    if total_samples > 0:
        overall_acc = total_tp / total_samples
    else:
        overall_acc = 0
        # Macro-F1
    if len(f1_scores) > 0:
        macro_f1 = np.mean(f1_scores)
    else:
        macro_f1 = 0.0
    values["loss"] = metrics["loss"] / num_samples_until_this_batch
    values["overall_acc"]=round(overall_acc,4)
    values["macro_f1"] = round(macro_f1, 4)
    return values
