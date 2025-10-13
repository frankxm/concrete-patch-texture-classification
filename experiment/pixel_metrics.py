# -*- coding: utf-8 -*-

"""
    The pixel metrics module
    ======================

    Use it to compute different metrics during evaluation.
    Available metrics:
        - Intersection-over-Union
        - Precision
        - Recall
        - F-score
"""

from shapely.geometry import Polygon, MultiPolygon,GeometryCollection
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

def compute_blackpixels(interpolygon,image,gt,pred):
    black_pixels = image == 0

    mask_structures=draw_polygon_mask(image, interpolygon)
    mask_gt=draw_polygon_mask(image,gt)
    mask_pred=draw_polygon_mask(image,pred)
    mask_combined = (mask_structures == 255) & black_pixels
    mask_combined_gt=(mask_gt == 255) & black_pixels
    mask_combined_pred = (mask_pred == 255) & black_pixels
    interpolygonimage = np.where(mask_combined, image, 255)
    # 计算区域内黑色像素的数量
    totalpixel_inter=np.sum(mask_combined)
    totalpixel_gt=np.sum(mask_combined_gt)
    totalpixel_pred=np.sum(mask_combined_pred)
    union=totalpixel_gt+totalpixel_pred-totalpixel_inter

    # plt.figure(2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image,cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(interpolygonimage,cmap='gray')

    plt.show()

    return totalpixel_inter,union,totalpixel_gt,totalpixel_pred
def draw_polygon_mask(image, polygons):
    mask_structures=np.zeros_like(image)

    if isinstance(polygons, MultiPolygon) or isinstance(polygons,GeometryCollection):
        for poly in polygons.geoms:
            if isinstance(poly, Polygon):
                intersection_points = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask_structures, [intersection_points.reshape((-1, 1, 2))], 255)
    elif isinstance(polygons, Polygon):
        intersection_points = np.array(polygons.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask_structures, [intersection_points.reshape((-1, 1, 2))], 255)

    return mask_structures
def compute_graypixels(gt,pred,imagegray):
    # 创建真实和预测多边形的掩码
    mask1 = draw_polygon_mask(imagegray, gt)
    mask2 = draw_polygon_mask(imagegray, pred)
    # 取对应mask的区域
    region1 = cv2.bitwise_and(imagegray, mask1)
    region2 = cv2.bitwise_and(imagegray, mask2)

    # 交集：取对应像素的最小值，代表共同的“最黑”内容
    intersection = np.minimum(region1, region2)
    # 并集：取对应像素的最大值，代表覆盖的“最亮”内容
    union = np.maximum(region1, region2)
    # 创建交集和并集掩码
    intersection_mask = (intersection > 0).astype(np.uint8)
    union_mask = (union > 0).astype(np.uint8)

    # 计算反灰度值（黑色内容），只在交集和并集区域内
    # 计算反灰度值（黑色内容）防止白色像素成为主导，因为关心黑色像素，如果直接加的话白色像素的值很大会影响
    # 这里是计算灰度级中黑色内容的总和，而不是黑色数量的总和 若只计算黑色内容像素的数量，可能会忽略灰度值的差异。它们的黑色内容强度不同。
    # 通过计算反灰度值的总和，可以综合考虑所有像素的黑色内容强度，而不仅仅是它们的数量。
    intersection_black_content = np.sum((255 - intersection) * intersection_mask)
    union_black_content = np.sum((255 - union) * union_mask)
    # plt.figure(1)
    # plt.subplot(2, 2, 1)
    # plt.imshow(region1, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(region2, cmap='gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(intersection,cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(union,cmap='gray')
    # plt.show()


    iou = intersection_black_content / union_black_content
    return intersection_black_content,union_black_content








def compute_intersection_union(gt, pred, imagegray, image):
    # 计算交集和并集像素数，避免重复计算
    inter_polygon = gt.intersection(pred)
    if inter_polygon.area != 0:
        # 计算交集的像素数量
        inter, union = compute_graypixels(gt, pred, imagegray)
        g, _ = compute_graypixels(gt, gt, imagegray)
        p, _ = compute_graypixels(pred, pred, imagegray)

        inter2, union2, g2, p2 = compute_blackpixels(inter_polygon, image, gt, pred)
        return inter, union, g, p, inter2, union2, g2, p2
    return None


def compute_metrics(labels, predictions, classes, global_metrics, image, imagergb, imagegray) -> dict:

    prediction_class = ["texte", "figure", "math"]

    # 使用多线程池来并行化计算
    with ThreadPoolExecutor() as executor:
        for channel in classes:
            t = time.time()
            inter_graypixel = 0
            union_graypixel = 0
            gt_graypixel = 0
            pred_graypixel = 0

            inter_blackpixel = 0
            union_blackpixel = 0
            gt_blackpixel = 0
            pred_blackpixel = 0

            futures = []
            for _, gt in labels[channel]:
                if channel not in prediction_class:
                    channel_pred = "math"
                else:
                    channel_pred = channel
                for _, pred in predictions[channel_pred]:
                    # 使用线程池来并行计算交集与并集
                    futures.append(executor.submit(compute_intersection_union, gt, pred, imagegray, image))

            # 处理线程返回结果
            for future in futures:
                result = future.result()
                if result is not None:
                    inter, union, g, p, inter2, union2, g2, p2 = result
                    inter_graypixel += inter
                    union_graypixel += union
                    gt_graypixel += g
                    pred_graypixel += p

                    inter_blackpixel += inter2
                    union_blackpixel += union2
                    gt_blackpixel += g2
                    pred_blackpixel += p2

            # 计算全局指标
            calculate_metrics(inter_blackpixel, union_blackpixel, gt_blackpixel, pred_blackpixel, global_metrics,
                              channel, image, 'black_')
            calculate_metrics(inter_graypixel, union_graypixel, gt_graypixel, pred_graypixel, global_metrics, channel,
                              image, 'gray_')
            print('当前类耗时：', time.time() - t)

    return global_metrics


def calculate_metrics(inter_area,union_area,gt_area,pred_area,global_metrics,channel,image,type):
    # 全局IOU，考虑所有标签和预测多边形的总的交集和并集，用于整体评估
    iou_global = get_iou(inter_area, union_area)
    global_metrics[channel][type+"iou"].append(iou_global)
    # 全局精确率，考虑所有标签和预测多边形的交集占所有预测多边形的比例
    precision = get_precision(inter_area, pred_area)
    # 全局召回率，考虑所有标签和预测多边形的交集占所有标签多边形的比例
    recall = get_recall(inter_area, gt_area)
    global_metrics[channel][type+"precision"].append(precision)
    global_metrics[channel][type+"recall"].append(recall)
    if precision + recall != 0:
        global_metrics[channel][type+"fscore"].append(
            2 * precision * recall / (precision + recall)
        )
    else:
        global_metrics[channel][type+"fscore"].append(0)
    dice=2*iou_global/(1+iou_global)
    global_metrics[channel][type+"dice"].append(dice)
#
# def compute_metrics(
#     labels: list, predictions: list, classes: list, global_metrics: dict,image,imagergb,imagegray
# ) -> dict:
#     t=time.time()
#     prediction_class=["texte", "figure", "math"]
#     for channel in classes:
#         inter_graypixel=0
#         union_graypixel=0
#         gt_graypixel=0
#         pred_graypixel=0
#
#         inter_blackpixel=0
#         union_blackpixel=0
#         gt_blackpixel=0
#         pred_blackpixel=0
#         for _, gt in labels[channel]:
#             if channel not in prediction_class:
#                 channel_pred="math"
#             else:
#                 channel_pred=channel
#             for _, pred in predictions[channel_pred]:
#                 interpolygon=gt.intersection(pred)
#                 if gt.intersection(pred).area!=0:
#                     inter,union=compute_graypixels(gt,pred,imagegray)
#                     g,g=compute_graypixels(gt,gt,imagegray)
#                     p,p=compute_graypixels(pred,pred,imagegray)
#                     inter_graypixel+=inter
#                     union_graypixel+=union
#                     gt_graypixel+=g
#                     pred_graypixel+=p
#
#                     inter2,union2,g2,p2=compute_blackpixels(interpolygon,image,gt,pred)
#                     inter_blackpixel+=inter2
#                     union_blackpixel+=union2
#                     gt_blackpixel+=g2
#                     pred_blackpixel+=p2
#
#         calculate_metrics(inter_blackpixel,union_blackpixel,gt_blackpixel,pred_blackpixel,global_metrics,channel,image,'black_')
#         calculate_metrics(inter_graypixel,union_graypixel,gt_graypixel,pred_graypixel,global_metrics,channel,image,'gray_')
#         print('当前类耗时：',time.time()-t)
#
#     return global_metrics

def compute_metrics_math(
    labels: list, predictions: list, classes: list, global_metrics: dict,image,imagergb,imagegray
) -> dict:

    inter_graypixel = 0
    union_graypixel = 0
    gt_graypixel = 0
    pred_graypixel = 0

    inter_blackpixel = 0
    union_blackpixel = 0
    gt_blackpixel = 0
    pred_blackpixel = 0
    for channel in classes:
        for _, gt in labels[channel]:
            for _, pred in predictions['math']:
                interpolygon=gt.intersection(pred)
                if gt.intersection(pred).area!=0:
                    inter,union=compute_graypixels(gt,pred,imagegray)
                    g,g=compute_graypixels(gt,gt,imagegray)
                    p,p=compute_graypixels(pred,pred,imagegray)
                    inter_graypixel+=inter
                    union_graypixel+=union
                    gt_graypixel+=g
                    pred_graypixel+=p

                    inter2,union2,g2,p2=compute_blackpixels(interpolygon,image,gt,pred)
                    inter_blackpixel+=inter2
                    union_blackpixel+=union2
                    gt_blackpixel+=g2
                    pred_blackpixel+=p2

    calculate_metrics(inter_blackpixel,union_blackpixel,gt_blackpixel,pred_blackpixel,global_metrics,'math_total',image,'black_')
    calculate_metrics(inter_graypixel,union_graypixel,gt_graypixel,pred_graypixel,global_metrics,'math_total',image,'gray_')

    return global_metrics

def get_iou(inter,union):
    if union == 0:
        return 1
    if inter == 0 and union != 0:
        return 0
    return inter / union

def get_dice(iou):

    return 2 * iou / (1 + iou)

def get_precision(intersection: float, predicted_area: float) -> float:
    """
    Get the precision between prediction and label areas of
    a given page.
    :param intersection: Area of the intersection.
    :param predicted_area: Area of the predicted objects.
    :return: The computed precision value.
    """
    # Nothing predicted.
    if predicted_area == 0:
        return 1
    return intersection / predicted_area


def get_recall(intersection, label_area) -> float:
    """
    Get the recall between prediction and label areas of
    a given page.
    :param intersection: Area of the intersection.
    :param label_area: Area of the label objects.
    :return: The computed recall value.
    """
    # Nothing to detect.
    if label_area == 0:
        return 1
    return intersection / label_area


