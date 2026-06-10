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

    cols_to_join = [c for c in ["class", "class2"] if c in df.columns]

    df_all = df_pred.join(
        df[cols_to_join],
        how="inner",
        rsuffix="_gt"
    )

    if filtered_label_evaluation is None:
        df_valid = df_all
        gt_labels = df_valid["class_gt"].values

        prediction_labels = df_valid["class"].values
    else:
        # 这里滤除filter_class in class1_gt，并且没有class2_gt的样本
        # 本质上是单标签
        mask_filtered = df_all["class_gt"] == filtered_label_evaluation
        mask_has_class2 = df_all["class2_gt"].notna()
        mask_keep = ~(mask_filtered & ~mask_has_class2)

        df_valid = df_all.loc[mask_keep]
        # 把gt_labels设为label2
        gt_labels = df_valid["class_gt"].where(
            df_valid["class_gt"] != filtered_label_evaluation,
            df_valid["class2_gt"]
        ).values

        prediction_labels = df_valid["class"].values

    from sklearn.utils.multiclass import unique_labels
    labels_in_subset = unique_labels(gt_labels, prediction_labels)
    labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]

    # 考虑多标签评估
    gt1 = df_valid["class_gt"].values
    gt2 = df_valid["class2_gt"].values if "class2_gt" in df_valid.columns else None
    pred = df_valid["class"].values
    strict_acc = compute_strict_accuracy(gt1, pred)
    tolerant_acc = compute_tolerant_accuracy(gt1, gt2, pred)
    soft_acc = compute_soft_accuracy(gt1, gt2, pred, w_secondary=0.5)


    metrics['strict_acc'] = round(strict_acc, 4)
    metrics['tolerant_acc'] = round(tolerant_acc, 4)
    metrics['soft_acc'] = round(soft_acc, 4)

    evaluate_subset("global", gt_labels, prediction_labels, labels_classes, metrics, evaluation_path)

    #
    # # #####不同视角/不同清晰度
    # eval_df = pd.DataFrame({
    #     "image": test_image_names,
    #     "gt": gt_labels,
    #     "pred": prediction_labels,
    #     "vue": [df.loc[img, "vue"] for img in test_image_names],
    #     "qualite": [df.loc[img, "qualite"] for img in test_image_names],
    # })
    #
    # for vue_value in eval_df["vue"].unique():
    #     sub_df = eval_df[eval_df["vue"] == vue_value]
    #     confusion_matrix_save_path=os.path.join(evaluation_path,f'{vue_value}')
    #     subset_gt = sub_df["gt"].values
    #     subset_pred = sub_df["pred"].values
    #     from sklearn.utils.multiclass import unique_labels
    #     labels_in_subset = unique_labels(subset_gt, subset_pred)
    #     labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]
    #     evaluate_subset(f"vue={vue_value}", subset_gt, subset_pred, labels_classes, metrics, confusion_matrix_save_path)
    #
    # for q_value in eval_df["qualite"].unique():
    #     sub_df = eval_df[eval_df["qualite"] == q_value]
    #     confusion_matrix_save_path=os.path.join(evaluation_path,f'{q_value}')
    #     subset_gt = sub_df["gt"].values
    #     subset_pred = sub_df["pred"].values
    #     from sklearn.utils.multiclass import unique_labels
    #     labels_in_subset = unique_labels(subset_gt, subset_pred)
    #     labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]
    #     evaluate_subset(f"qualite={q_value}", subset_gt, subset_pred, labels_classes, metrics, confusion_matrix_save_path)
    #
    #




    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished evaluating in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )


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
    print("Strict Accuracy (label1 only):      ", metrics['strict_acc'] )
    print("Tolerant Accuracy (label1 or 2):    ", metrics['tolerant_acc'])
    print("Soft Accuracy (label1=1, label2=0.5):", metrics['soft_acc'])




    ######## 不同accuracy指标分析(多预测，多label)top1-accuracy_label1,top2-accuracy-label1,top1-accuracy-label2,top2-accuracy_label2,top1-accuracy_deuxlabel,top2-accuracy-deuxlabel
    # gt_labels2 = np.array([
    #     df.loc[img, "class2"] if pd.notna(df.loc[img, "class2"])
    #     else df.loc[img, "class"]
    #     for img in test_image_names
    # ])
    #
    # prediction_labels2 = df_prediction["class2"].values
    # top2_correct = (gt_labels == prediction_labels) | (gt_labels == prediction_labels2)
    # top2_acc = np.mean(top2_correct)
    # print("------Top-k Accuracy------")
    # print(f"Top-1 Accuracy: {np.mean(gt_labels == prediction_labels):.4f}")
    # print(f"Top-2 Accuracy: {top2_acc:.4f}")
    # print("------2em label------")
    # top2_correct = (gt_labels2 == prediction_labels) | (gt_labels2 == prediction_labels2)
    # top2_acc = np.mean(top2_correct)
    # print(f"Top-1 Accuracy: {np.mean(gt_labels2 == prediction_labels):.4f}")
    # print(f"Top-2 Accuracy: {top2_acc:.4f}")
    #
    # print("------Multi label(2 classes)------")
    # top_acc_multilabel = (gt_labels == prediction_labels) | (gt_labels2 == prediction_labels)
    # print(f"Top-1 Accuracy: {np.mean(top_acc_multilabel):.4f}")
    #
    # top2_acc_multilabel = (gt_labels == prediction_labels) | (gt_labels2 == prediction_labels)|(gt_labels == prediction_labels2) | (gt_labels2 == prediction_labels2)
    # print(f"Top-2 Accuracy: {np.mean(top2_acc_multilabel):.4f}")

    ev_utils.save_results(
        metrics,
        ground_classes_names,
        confusion_matrix_savepath,name
    )


def compute_strict_accuracy(gt1, pred):
    return (pred == gt1).mean()


def compute_tolerant_accuracy(gt1, gt2, pred):
    if gt2 is None:
        return (pred == gt1).mean()

    gt2_valid = ~pd.isna(gt2)
    correct = (pred == gt1) | (gt2_valid & (pred == gt2))
    return correct.mean()


def compute_soft_accuracy(gt1, gt2, pred, w_secondary=0.5):
    score = np.zeros(len(pred), dtype=float)

    # 主标签命中
    score[pred == gt1] = 1.0

    if gt2 is not None:
        gt2_valid = ~pd.isna(gt2)
        mask = (pred == gt2) & (pred != gt1) & gt2_valid
        score[mask] = w_secondary

    return score.mean()