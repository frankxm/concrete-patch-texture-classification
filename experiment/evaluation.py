# -*- coding: utf-8 -*-

"""
    The evaluation utils module
    ======================

    Use it to during the evaluation stage.
"""

import json
import os
import warnings

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from shapely import MultiPolygon
from shapely.geometry import Polygon
import seaborn as sns
import copy
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
# 忽略所有的 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)

def save_results(
        metrics,ground_classes_names,evaluation_path,description
):


    json_dict = {"description": description}

    for channel in ground_classes_names:
        json_dict[channel] = {}
        json_dict[channel]["precision"] =metrics[channel]["precision"]
        json_dict[channel]["recall"] = metrics[channel]["recall"]
        json_dict[channel]["fscore"] = metrics[channel]["fscore"]
    json_dict['overall_acc'] =metrics['overall_acc']
    json_dict['weighted_precision'] = metrics['weighted_precision']
    json_dict["weighted_recall"] = metrics["weighted_recall"]
    json_dict["weighted_f1"] = metrics["weighted_f1"]
    json_dict["macro_precision"] = metrics["macro_precision"]
    json_dict["macro_recall"] = metrics["macro_recall"]
    json_dict["macro_f1"] = metrics["macro_f1"]
    json_dict["micro_precision"] = metrics["micro_precision"]
    json_dict["micro_recall"] = metrics["micro_recall"]
    json_dict["micro_f1"] = metrics["micro_f1"]


    with open(os.path.join(evaluation_path,"eval_results.json"), "w") as json_file:
        json.dump(json_dict, json_file, indent=4)



def plot_confusion_matrix(cm_list,type_list, ground_classes_names, evaluation_path):
    for index,cm in enumerate(cm_list):
        type=type_list[index]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ground_classes_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title("Confusion Matrix")
        save_path = os.path.join(evaluation_path, f"confusion_matrix_{type}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()


# Macro-f1适合类别不平衡，每个类别都考虑。关注小样本准确率用Macro
# Micro-f1偏向多数类，适合类别平衡。

# 1.各个类别下样本数量不一样多但相差不大  那在这情况下，就是用macro-f1与micro-f1都行。
# 但是当数据中存在某类f1值较大，有的类f1很小，在这种情况下，macro-f1这时候就明显会受到某类f1小的值影响，会偏小。类别分布不平衡时，强调每个类都同等重要。对小类敏感，不会被大类掩盖。在样本极度不平衡时可能偏低。
# 2.当各类别下数据量不一样多但相差很大   这种情况下，也就是数据极度不均衡的情况下，micro-f1影响就很大，micro-f1此时几乎只能反映那个类样本量大的情况，micro-f1≈A类f1。关心总体预测能力，而不关心每一类的单独表现。
# 3.Weighted F1 对每个类别计算 F1 分数后，按该类别的样本数加权平均。适用情况：类别不平衡时，考虑每类样本的重要性。优点：综合性能评估准确，不偏向小类或大类。

# 类别不平衡时 Micro F1 ≥ Weighted F1 ≥ Macro F1
def compute_metrics(matrice,classes,values):

    total_tp = 0
    total_samples = 0
    for channel in classes:
        recall, precision, f1 = compute(matrice, classes.index(channel))
        values[channel]['recall'] = round(recall, 4)
        values[channel]['precision'] = round(precision, 4)
        values[channel]['fscore'] = round(f1, 4)

        # 累计总的TP和所有样本数
        total_tp += matrice[classes.index(channel), classes.index(channel)]

        total_samples += np.sum(matrice[classes.index(channel), :])

        # 计算所有类的总准确率
    if total_samples > 0:
        overall_acc = total_tp / total_samples

    else:
        overall_acc = 0
    values["overall_acc"]=round(overall_acc,4)
    return values

def compute(confusion_matrix: np.ndarray, channel: str) -> float:

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

def compute_macro_weighted_micro(y_true,y_pred,metrics):

    # print('Accuracy', accuracy_score(y_true, y_pred))
    # print('------Weighted------')
    # print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
    # print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
    # print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
    metrics['weighted_precision']=round(precision_score(y_true, y_pred, average='weighted'), 4)

    metrics['weighted_recall'] =round(recall_score(y_true, y_pred, average='weighted'),4)
    metrics['weighted_f1'] = round(f1_score(y_true, y_pred, average='weighted'),4)

    metrics['macro_precision'] = round(precision_score(y_true, y_pred, average='macro'),4)
    metrics['macro_recall'] = round(recall_score(y_true, y_pred, average='macro'),4)
    metrics['macro_f1'] = round(f1_score(y_true, y_pred, average='macro'),4)

    metrics['micro_precision'] = round(precision_score(y_true, y_pred, average='micro'),4)
    metrics['micro_recall'] = round(recall_score(y_true, y_pred, average='micro'),4)
    metrics['micro_f1'] = round(f1_score(y_true, y_pred, average='micro'),4)

    # print('------Macro------')
    # print('Macro precision', precision_score(y_true, y_pred, average='macro'))
    # print('Macro recall', recall_score(y_true, y_pred, average='macro'))
    # print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
    #
    # print('------Micro------')
    # print('Micro precision', precision_score(y_true, y_pred, average='micro'))
    # print('Micro recall', recall_score(y_true, y_pred, average='micro'))
    # print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))
    return metrics