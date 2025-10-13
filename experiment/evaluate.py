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
from tqdm import tqdm

import evaluation as ev_utils
import object_metrics as o_metrics
import pixel_metrics as p_metrics
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score


def run(
    log_path: str,
    classes_names: list,
    set: str,
    label_path: Path,
    image_path: Path,
    evaluation_path: Path,
    prediction_label_path
):

    # Run evaluation.

    starting_time = time.time()

    label_dir = label_path
    image_dir = image_path

    ground_classes_names = classes_names

    metrics= {
        channel: {metric: {} for metric in ["precision", "recall", "fscore"]}
        for channel in ground_classes_names
    }


    num_classes = len(ground_classes_names)


    df = pd.read_csv(label_path['label'])
    df["image_base"] = df["image"].apply(lambda x: os.path.splitext(x)[0])
    df.set_index("image_base", inplace=True)


    prediction_label_path=os.path.join(prediction_label_path,'predictions.csv')
    df_prediction = pd.read_csv(prediction_label_path)
    #正常情况，只用一列的class
    prediction_labels = df_prediction["class"].values
    # ecrase预测过多时，用于测试模型泛化性时，用其他数据集时，把预测的ecrase类换成class2
    # prediction_labels = df_prediction["class"].copy()
    # prediction_labels[df_prediction["class"] == 4] = df_prediction["class2"][df_prediction["class"] == 4]
    # prediction_labels = prediction_labels.values


    test_image_names = df_prediction["image"].values
    if test_image_names[0].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        test_image_names = [img.split('.')[0] for img in test_image_names]

    # 当训练的模型只能区分四类时，正常情况/五类时正常情况
    gt_labels = np.array([df.loc[img, "class"] for img in test_image_names])

    # 训练的模型只能区分四类时，把label中ecrase的类换成class2
    # gt_labels = np.array([
    #     df.loc[img, "class2"] if df.loc[img, "class"] == 4 and pd.notna(df.loc[img, "class2"])
    #     else df.loc[img, "class"]
    #     for img in test_image_names
    # ])
    # 使用class2
    # gt_labels = np.array([
    #     df.loc[img, "class2"] if pd.notna(df.loc[img, "class2"])
    #     else df.loc[img, "class"]
    #     for img in test_image_names
    # ])

    from sklearn.utils.multiclass import unique_labels
    labels_in_subset = unique_labels(gt_labels, prediction_labels)
    labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]
    evaluate_subset("global", gt_labels, prediction_labels,labels_classes,metrics,evaluation_path)


    eval_df = pd.DataFrame({
        "image": test_image_names,
        "gt": gt_labels,
        "pred": prediction_labels,
        "vue": [df.loc[img, "vue"] for img in test_image_names],
        "qualite": [df.loc[img, "qualite"] for img in test_image_names],
    })
    #####不同视角/不同清晰度
    for vue_value in eval_df["vue"].unique():
        sub_df = eval_df[eval_df["vue"] == vue_value]
        confusion_matrix_save_path=os.path.join(evaluation_path,f'{vue_value}')
        subset_gt = sub_df["gt"].values
        subset_pred = sub_df["pred"].values
        from sklearn.utils.multiclass import unique_labels
        labels_in_subset = unique_labels(subset_gt, subset_pred)
        labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]
        evaluate_subset(f"vue={vue_value}", subset_gt, subset_pred, labels_classes, metrics, confusion_matrix_save_path)

    # # 分 qualite 评估

    for q_value in eval_df["qualite"].unique():
        sub_df = eval_df[eval_df["qualite"] == q_value]
        confusion_matrix_save_path=os.path.join(evaluation_path,f'{q_value}')
        subset_gt = sub_df["gt"].values
        subset_pred = sub_df["pred"].values
        from sklearn.utils.multiclass import unique_labels
        labels_in_subset = unique_labels(subset_gt, subset_pred)
        labels_classes = [ground_classes_names[int(i)] for i in labels_in_subset]
        evaluate_subset(f"qualite={q_value}", subset_gt, subset_pred, labels_classes, metrics, confusion_matrix_save_path)






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
