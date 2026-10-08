# -*- coding: utf-8 -*-

"""
    The evaluation module
    ======================

    Use it to evaluation a trained network.
"""

import logging
import os
import time
from pathlib import Path
import numpy as np
import evaluation as ev_utils
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

def run(
    ground_classes_names: list,
    label_path: Path,
    evaluation_path: Path,
    prediction_label_path,
filtered_label_evaluation
):
    metrics = {
    channel: {metric: {} for metric in ["precision", "recall", "fscore"]}
    for channel in ground_classes_names}

    starting_time = time.time()

    # gt
    df = pd.read_csv(label_path['label'])
    df["image_base"] = df["image"].apply(lambda x: os.path.splitext(x)[0])
    df.set_index("image_base", inplace=True)
    # prediction
    prediction_label_path = os.path.join(prediction_label_path, 'predictions.csv')
    df_pred = pd.read_csv(prediction_label_path)
    df_pred["image_base"] = df_pred["image"].apply(lambda x: os.path.splitext(x)[0])
    df_pred.set_index("image_base", inplace=True)

    cols_to_join = [c for c in ["class", "class2","vue","qualite"] if c in df.columns]

    df_all = df_pred.join(
        df[cols_to_join].add_suffix("_gt"),
        how="inner"
    )




    evaluate_subset_df(df_all,'global',metrics,ground_classes_names,evaluation_path,filtered_label_evaluation)
    ######视角，清晰度，用于terre
    #
    # for v in ["dessus", "laterale"]:
    #     df_v = df_all[df_all["vue"] == v]
    #     if len(df_v) > 0:
    #         evaluate_subset_df(df_v, f"vue_{v}",metrics,ground_classes_names,evaluation_path,filtered_label_evaluation)
    #
    # for q in ["flou", "clair"]:
    #     df_q = df_all[df_all["qualite"] == q]
    #     if len(df_q) > 0:
    #         evaluate_subset_df(df_q, f"qualite_{q}",metrics,ground_classes_names,evaluation_path,filtered_label_evaluation)



    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished evaluating in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )

def evaluate_subset_df(df_subset, name,metrics,ground_classes_names,evaluation_path,filtered_label_evaluation):

    if filtered_label_evaluation is None:
        df_valid = df_subset
        gt_labels = df_valid["class_gt"].values
    else:
        # 这里滤除filter_class in class1_gt，并且没有class2_gt的样本
        # 本质上是单标签
        mask_filtered = df_subset["class_gt"] == filtered_label_evaluation
        mask_has_class2 = df_subset["class2_gt"].notna()
        mask_keep = ~(mask_filtered & ~mask_has_class2)

        df_valid = df_subset.loc[mask_keep]
        # 把gt_labels设为label2
        gt_labels = df_valid["class_gt"].where(
            df_valid["class_gt"] != filtered_label_evaluation,
            df_valid["class2_gt"]
        ).values

    prediction_labels = df_valid["class"].values
    if "class2" in df_valid.columns:
        prediction_labels2=df_valid["class2"].values
        no_prediction_class2=0
    else :
        no_prediction_class2=1

    # 考虑多标签评估label1, label2
    gt2 = df_valid["class2_gt"].values if "class2_gt" in df_valid.columns else None
    strict_acc,acc_nogt2,accgt2 = compute_strict_accuracy(gt_labels, gt2,prediction_labels)
    tolerant_acc,tolacc_gt2 = compute_tolerant_accuracy(gt_labels, gt2, prediction_labels)
    soft_acc = compute_soft_accuracy(gt_labels, gt2, prediction_labels, w_secondary=0.5)

    metrics['strict_acc'] = round(strict_acc, 4)
    metrics['acc_nogt2'] = round(acc_nogt2, 4)
    metrics['acc_gt2']   = round(accgt2, 4)
    metrics['tolerant_acc'] = round(tolerant_acc, 4)
    metrics['tolerant_acc_gt2'] = round(tolacc_gt2, 4)
    metrics['soft_acc'] = round(soft_acc, 4)

    if not no_prediction_class2:

        top2_all, top2_nogt2, top2_gt2= compute_top2acc(gt_labels, gt2, prediction_labels,prediction_labels2)

        metrics['top2_acc_all'] = round(top2_all, 4)
        metrics['top2_acc_nogt2'] = round(top2_nogt2, 4)
        metrics['top2_acc_gt2'] = round(top2_gt2, 4)

    # 整体评估
    from sklearn.utils.multiclass import unique_labels
    labels_in_subset = unique_labels(gt_labels, prediction_labels)
    labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]

    evaluate_subset(name, gt_labels, prediction_labels, labels_classes, metrics, os.path.join(evaluation_path,name))




def evaluate_subset(name, subset_gt, subset_pred,ground_classes_names,metrics,confusion_matrix_savepath):
    if not os.path.exists(confusion_matrix_savepath):
        os.makedirs(confusion_matrix_savepath)
    print(f"\n===== evaluation: {name} =====")
    metrics_local = ev_utils.compute_macro_weighted_micro(subset_gt, subset_pred, metrics)
    cm = confusion_matrix(subset_gt, subset_pred)


    mcc = matthews_corrcoef(subset_gt, subset_pred)
    metrics['mcc']=round(mcc, 4)
    cm_true_normalized = confusion_matrix(subset_gt, subset_pred, normalize='true')
    cm_pred_normalized = confusion_matrix(subset_gt, subset_pred, normalize='pred')
    cm_list = [cm, cm_true_normalized, cm_pred_normalized]
    type_list = ['original', 'true_normalized', 'pred_normalized']
    ev_utils.plot_confusion_matrix(cm_list, type_list, ground_classes_names, confusion_matrix_savepath)
    metrics_local = ev_utils.compute_metrics(cm, ground_classes_names, metrics_local)

    for channel in ground_classes_names:
        print(channel)
        print(f"Precision       = ", metrics_local[channel]["precision"])
        print(f"Recall          = ", metrics_local[channel]["recall"])
        print(f"Fscore          = ", metrics_local[channel]["fscore"])
        print("\n")
    print('Accuracy', metrics_local['overall_acc'])
    print('MCC', metrics_local['mcc'])
    print('------Weighted------')
    print('Weighted precision', metrics_local['weighted_precision'])
    print('Weighted recall', metrics_local['weighted_recall'])
    print('Weighted f1-score', metrics_local['weighted_f1'])
    print('------Macro------')
    print('Macro precision', metrics_local['macro_precision'])
    print('Macro recall', metrics_local['macro_recall'])
    print('Macro f1-score', metrics_local['macro_f1'])
    print('------Micro------')
    print('Micro precision', metrics_local['micro_precision'])
    print('Micro recall', metrics_local['micro_recall'])
    print('Micro f1-score', metrics_local['micro_f1'])

    print("\n===== Custom Accuracy Metrics =====")
    print("Strict Accuracy (label1 only)all samples:      ", metrics['strict_acc'] )
    print("Strict Accuracy no gt2 samples:      ", metrics['acc_nogt2'])
    print("Strict Accuracy with gt2 samples:      ", metrics['acc_gt2'])
    print("Tolerant Accuracy (label1 or 2) all samples:    ", metrics['tolerant_acc'])
    print("Tolerant Accuracy (label1 or 2) with gt2 samples:    ", metrics['tolerant_acc_gt2'])
    print("Soft Accuracy (label1=1, label2=0.5):", metrics['soft_acc'])


    print("Top-2 Accuracy all samples:      ", metrics['top2_acc_all'])
    print("Top-2 Accuracy without gt2 samples:      ", metrics['top2_acc_nogt2'])
    print("Top-2 Accuracy with gt2 samples:      ", metrics['top2_acc_gt2'])





    ev_utils.save_results(
        metrics,
        ground_classes_names,
        confusion_matrix_savepath,name
    )


def compute_strict_accuracy(gt1, gt2,pred):
    correct_all_samples = (pred == gt1)
    if gt2 is not None:
        gt2_valid = ~pd.isna(gt2)
        acc_nogt2 = (pred[~gt2_valid] == gt1[~gt2_valid]).mean()
        acc_gt2 = (pred[gt2_valid] == gt1[gt2_valid]).mean()
        return correct_all_samples.mean(),acc_nogt2,acc_gt2
    else:
        return correct_all_samples.mean(),-1,-1


def compute_tolerant_accuracy(gt1, gt2, pred):
    if gt2 is None:
        return -1,-1

    gt2_valid = ~pd.isna(gt2)
    correct = (pred == gt1) | (gt2_valid & (pred == gt2))
    correct_gt2= correct[gt2_valid]
    return correct.mean(),correct_gt2.mean()

def compute_top2acc(gt, gt2,pred, pred2):
    top2_all = (gt == pred) | (gt == pred2)
    if gt2 is not None:
        gt2_valid = ~pd.isna(gt2)
        # no gt2 subset
        top2_nogt2 = top2_all[~gt2_valid].mean() if (~gt2_valid).sum() > 0 else np.nan
        # gt2 subset
        top2_gt2 = top2_all[gt2_valid].mean() if gt2_valid.sum() > 0 else np.nan

        return top2_all.mean(), top2_nogt2, top2_gt2
    else:
        return top2_all.mean(),-1,-1



def compute_soft_accuracy(gt1, gt2, pred, w_secondary=0.5):
    score = np.zeros(len(pred), dtype=float)

    # 主标签命中
    score[pred == gt1] = 1.0

    if gt2 is not None:
        gt2_valid = ~pd.isna(gt2)
        mask = (pred == gt2) & (pred != gt1) & gt2_valid
        score[mask] = w_secondary

    return score.mean()