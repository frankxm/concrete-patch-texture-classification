# -*- coding: utf-8 -*-

"""
    The object metrics module
    ======================

    Use it to compute different metrics during evaluation.
    Available metrics:
        - Precision
        - Recall
        - F-score
        - Average precision
"""
import time

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from pixel_metrics import compute_graypixels,compute_blackpixels,get_iou,get_dice
import cv2
import concurrent.futures

import matplotlib.pyplot as plt
import os
import seaborn as sns

def __compute_black_dice_from_contours(poly1, poly2, image) :

    intersection_polygon = poly1.intersection(poly2)

    totalpixel_inter,union,totalpixel_gt,totalpixel_pred = compute_blackpixels(intersection_polygon,image,poly1,poly2)
    iou=get_iou(totalpixel_inter,union)
    dice=get_dice(iou)
    return dice

def __compute_gray_dice_from_contours(poly1,poly2,imagegray):
    inter, union = compute_graypixels(poly1, poly2, imagegray)
    iou=get_iou(inter, union)
    dice=get_dice(iou)
    return dice










def __compute_black_iou_from_contours(poly1, poly2, image):

    intersection_polygon = poly1.intersection(poly2)

    totalpixel_inter,union,totalpixel_gt,totalpixel_pred = compute_blackpixels(intersection_polygon,image,poly1,poly2)
    iou=get_iou(totalpixel_inter,union)
    return iou

def __compute_gray_iou_from_contours(poly1,poly2,imagegray):
    inter, union = compute_graypixels(poly1, poly2, imagegray)
    iou=get_iou(inter, union)
    return iou



def __get_ious(labels: list, predictions: list, image,type) :

    ious = {key: 0 for key in range(len(predictions))}
    # 使用 ThreadPoolExecutor 创建一个线程池，以便可以并行地执行多个计算任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of future objects
        futures = []
        for _, label in labels:
            for index, prediction in enumerate(predictions):
                if label.intersection(prediction[1]).area != 0:
                    # executor.submit(...)：将计算任务提交到线程池，submit 方法返回一个 Future 对象，代表异步执行的任务。
                    if type=='black' :
                        future = executor.submit(__compute_black_iou_from_contours,label, prediction[1],image)
                    elif type=='gray':
                        future=executor.submit(__compute_gray_iou_from_contours,label,prediction[1],image)
                    futures.append((future, index))

        # 保证对每个预测，保存它与所有标签计算出来的 IoU 中的最大值。
        for future, index in futures:
            iou = future.result()
            if iou > ious[index]:
                ious[index] = iou

    return ious
def __get_dice(labels: list, predictions: list, image,type) -> dict:

    dice_final = {key: {'iou':0,'coverage_ratio':0} for key in range(len(predictions))}
    # 使用 ThreadPoolExecutor 创建一个线程池，以便可以并行地执行多个计算任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of future objects
        futures = []
        for _, label in labels:
            for index, prediction in enumerate(predictions):
                if label.intersection(prediction[1]).area != 0:
                    # executor.submit(...)：将计算任务提交到线程池，submit 方法返回一个 Future 对象，代表异步执行的任务。
                    if type=='black' :
                        future = executor.submit(__compute_black_dice_from_contours,label, prediction[1],image)
                    elif type=='gray':
                        future=executor.submit(__compute_gray_dice_from_contours,label,prediction[1],image)
                    intersection=label.intersection(prediction[1])
                    coverage_ratio=intersection.area/label.area
                    futures.append((future, index, coverage_ratio))


        for future, index,coverage_ratio in futures:
            dice = future.result()
            if dice > dice_final[index]['iou']:
                dice_final[index]['iou'] = dice
                dice_final[index]['coverage_ratio'] = coverage_ratio

    return dice_final
def __rank_predicted_objects(labels: list, predictions: list, image,imagegray) -> dict:

    black_ious = __get_ious(labels, predictions, image,'black')
    gray_ious= __get_ious(labels,predictions,imagegray,'gray')
    # 取出每个预测多边形的置信度分数，和它的IOU
    scores = {index: prediction[0] for index, prediction in enumerate(predictions)}
    tuples_score_iou_black = [(v, black_ious[k]) for k, v in scores.items()]
    scores_black = sorted(tuples_score_iou_black, key=lambda item: (-item[0], -item[1]))
    tuples_score_iou_gray = [(v, gray_ious[k]) for k, v in scores.items()]
    scores_gray = sorted(tuples_score_iou_gray, key=lambda item: (-item[0], -item[1]))
    return scores_black,scores_gray
def __rank_predicted_objects_dice(labels: list, predictions: list, image,imagegray) -> dict:

    black_dice = __get_dice(labels, predictions, image, 'black')
    gray_dice = __get_dice(labels, predictions, imagegray, 'gray')
    # 取出每个预测多边形的置信度分数，和它的IOU
    scores = {index: prediction[0] for index, prediction in enumerate(predictions)}
    tuples_score_iou_black = [(v, black_dice[k]) for k, v in scores.items()]
    scores_black = sorted(tuples_score_iou_black, key=lambda item: (-item[0]))
    tuples_score_iou_gray = [(v, gray_dice[k]) for k, v in scores.items()]
    scores_gray = sorted(tuples_score_iou_gray, key=lambda item: (-item[0]))
    return scores_black,scores_gray
def compute(chanel_scores,scores,channel):
    for iou in range(50, 100, 5):
        rank_scores = {rank: {"True": 0, "Total": 0} for rank in range(95, -5, -5)}
        for rank in range(95, -5, -5):
            rank_objects = list(
                filter(lambda item: item[0] >= rank / 100, chanel_scores)
            )

            tp_count=0
            pred_count= len(rank_objects)
            for prediction in rank_objects:

                iou_value = prediction[1]['iou']
                coverage_ratio = prediction[1]['coverage_ratio']

                if iou_value > iou / 100 or coverage_ratio>0.8:
                    tp_count+=1
            rank_scores[rank]["True"] = tp_count
            rank_scores[rank]["Total"] = pred_count
        scores[channel][iou] = rank_scores
    return scores
def compute_rank_scores(labels: list, predictions: list, classes: list, image,imagegray) -> dict:
    """
    对于预测的每一类，计算该类下所有预测多边形的iou值和置信度分数打包。之后在不同的iou阈值和置信度阈值下统计TP数量
                    不同的iou阈值是因为目标级的指标，不同的置信度阈值是要在当前iou下画出P-R曲线，iou作用是区分tp，置信度作用是区分有多少预测物体
                    最后得到的scores分数是每个类下每个IOU阈值下每个置信度阈值下的TP数量和总数量的字典
    """
    prediction_class = ["texte", "figure", "math"]
    scores_black = {channel: {iou: None for iou in range(50, 100, 5)} for channel in classes}
    scores_gray={channel: {iou: None for iou in range(50, 100, 5)} for channel in classes}
    for channel in classes:
        if channel not in prediction_class:
            channel_pred = "math"
        else:
            channel_pred = channel
        channel_scores_black,channel_scores_gray = __rank_predicted_objects_dice(labels[channel], predictions[channel_pred], image,imagegray)
        scores_black=compute(channel_scores_black,scores_black,channel)
        scores_gray=compute(channel_scores_gray,scores_gray,channel)

    return scores_black,scores_gray


def update_rank_scores(global_scores: dict, image_scores: dict, classes: list) -> dict:
    """
    Update the global scores by adding page scores.
    :param global_scores: The scores obtained so far.
    :param image_scores: the current page scores.
    :param classes: The classes names involved in the experiment.
    :return global_scores: The updated global scores.
    """
    for channel in classes:
        for iou in range(50, 100, 5):
            for rank in range(95, -5, -5):
                global_scores[channel][iou][rank]["True"] += image_scores[channel][iou][
                    rank
                ]["True"]
                global_scores[channel][iou][rank]["Total"] += image_scores[channel][
                    iou
                ][rank]["Total"]
    return global_scores


def __init_results() -> dict:
    """
    Initialize the results dictionary by generating dictionary for
    the different rank and Intersection-over-Union thresholds.
    :return: The initialized results dictionary.
    """
    return {iou: {rank: 0 for rank in range(95, -5, -5)} for iou in range(50, 100, 5)}


def __get_average_precision(precisions: list, recalls: list) -> float:
    """
    Compute the mean average precision. Interpolate the precision-recall
    curve, then get the interpolated precisions for values.
    Compute the average precision.
    :param precisions: The computed precisions for a given channel and a
                       given confidence score.
    :param recalls: The computed recalls for a given channel and a given
                    confidence score.
    :return: The average precision for the channel and for the confidence
             score range.
    """
    rp_tuples = []
    # Interpolated precision-recall curve.
    while len(precisions) > 0:
        max_precision = np.max(precisions)
        argmax_precision = np.argmax(precisions)
        max_recall = recalls[argmax_precision]
        rp_tuples.append({"p": max_precision, "r": max_recall})
        for _ in range(argmax_precision + 1):
            precisions.pop(0)
            recalls.pop(0)
    rp_tuples[-1]["r"] = 1
    # 曲线在召回率维度上能够从0开始插值到1，同时确保精度从最高值开始逐步降低
    ps = [rp_tuple["p"] for rp_tuple in rp_tuples]
    rs = [rp_tuple["r"] for rp_tuple in rp_tuples]
    ps.insert(0, ps[0])
    rs.insert(0, 0)

    # 梯形法则是一种数值积分方法，它通过将曲线下的区域分成多个梯形来估算曲线下的面积
    return np.trapz(ps, x=rs)

def compute_confusion_matrix(channel,true_predicted,predicted,true_gt,iou,rank,iou_list,confidence_list,iou_weights,confidence_weights,classes,weighted_confusion_matrix):
    if (iou / 100 in iou_list) and (rank / 100 in confidence_list):
        iou_weight = iou_weights[iou_list.index(iou / 100)]
        conf_weight = confidence_weights[confidence_list.index(rank / 100)]
        confusion_matrices = np.zeros((len(classes), len(classes)), dtype=float)
        # 仅在有预测时更新混淆矩阵
        if predicted > 0:
            # 正确预测
            confusion_matrices[classes.index(channel)][classes.index(channel)] += true_predicted
            # 计算错误预测 FP
            false_predicted = predicted - true_predicted if predicted != 0 else 1
            # 将错误的预测计入混淆矩阵
            for other_channel in classes:
                if other_channel != channel:
                    confusion_matrices[classes.index(other_channel)][classes.index(channel)] += false_predicted // (
                                len(classes) - 1)
            # 计算漏检 FN
            no_predicted = true_gt[channel] - true_predicted if true_gt[channel] != 0 else 1
            for other_channel in classes:
                if other_channel != channel:
                    confusion_matrices[classes.index(channel)][classes.index(other_channel)] += no_predicted // (
                                len(classes) - 1)
        weighted_confusion_matrix += iou_weight * conf_weight * confusion_matrices
    return weighted_confusion_matrix


def get_mean_results(global_scores, true_gt, classes, results) :
    iou_list = [0.5, 0.7, 0.9]
    iou_weights = [0.5, 0.3, 0.2]

    confidence_list = [0.2, 0.5, 0.8]
    confidence_weights = [0.2, 0.5, 0.3]

    list1=["texte", "figure", "math", "mathstructuree", "textemath", "mathbarree"]
    weighted_confusion_matrix = np.zeros((6, 6), dtype=float)
    list2 = ["texte", "figure", "math_total"]
    weighted_confusion_matrix_combined= np.zeros((3, 3), dtype=float)
    for channel in classes:
        precisions = __init_results()
        recalls = __init_results()
        fscores = __init_results()
        aps = {iou: 0 for iou in range(50, 100, 5)}
        for iou in range(50, 100, 5):
            for rank in range(95, -5, -5):

                true_predicted = global_scores[channel][iou][rank]["True"]
                predicted = global_scores[channel][iou][rank]["Total"]
                if channel in list1:
                    weighted_confusion_matrix=compute_confusion_matrix(channel,true_predicted,predicted,true_gt,iou,rank,iou_list,confidence_list,iou_weights,confidence_weights,list1,weighted_confusion_matrix)
                if channel in list2:
                    weighted_confusion_matrix_combined=compute_confusion_matrix(channel,true_predicted, predicted, true_gt,iou,rank,iou_list,confidence_list,iou_weights,confidence_weights,list2,weighted_confusion_matrix_combined)


                precisions[iou][rank] = (
                    true_predicted / predicted if predicted != 0 else 0
                )
                recalls[iou][rank] = (
                    true_predicted / true_gt[channel] if true_gt[channel] != 0 else 0
                )

                if precisions[iou][rank] + recalls[iou][rank] != 0:
                    fscores[iou][rank] = (
                            2
                            * (precisions[iou][rank] * recalls[iou][rank])
                            / (precisions[iou][rank] + recalls[iou][rank])
                    )
            aps[iou] = __get_average_precision(
                list(precisions[iou].values()), list(recalls[iou].values())
            )
            results[channel]["precision"] = precisions
            results[channel]["recall"] = recalls
            results[channel]["fscore"] = fscores
            results[channel]["AP"] = aps

    return results,weighted_confusion_matrix,weighted_confusion_matrix_combined


