# -*- coding: utf-8 -*-

"""
    The prediction utils module
    ======================

    Use it to during the prediction stage.
"""

import json

import cv2
import imageio as io
import numpy as np
import os
from shapely.geometry import Polygon
def resize_polygons(
    polygons: dict, image_size: tuple, input_size: tuple, padding: tuple
) -> dict:

    # Compute the small size image.
    ratio = float(input_size) / max(image_size)
    new_size = tuple([int(x * ratio) for x in image_size])
    # Compute resizing ratio
    ratio = [element / float(new) for element, new in zip(image_size, new_size)]


    for channel in polygons.keys():
        for index, polygon in enumerate(polygons[channel]):
            x_points = [element[0][1] for element in polygon["polygon"]]
            y_points = [element[0][0] for element in polygon["polygon"]]
            # 只减去顶部和左侧填充，因为这两部分影响着坐标 底部和右侧填充不影响坐标，只是让图像尺寸变大
            x_points = [
                int((element - padding["top"]) * ratio[0]) for element in x_points
            ]
            y_points = [
                int((element - padding["left"]) * ratio[1]) for element in y_points
            ]

            x_points = [
                int(element) if element < image_size[0] else int(image_size[0])
                for element in x_points
            ]
            y_points = [
                int(element) if element < image_size[1] else int(image_size[1])
                for element in y_points
            ]
            x_points = [int(element) if element > 0 else 0 for element in x_points]
            y_points = [int(element) if element > 0 else 0 for element in y_points]
            assert max(x_points) <= image_size[0]
            assert min(x_points) >= 0
            assert max(y_points) <= image_size[1]
            assert min(y_points) >= 0
            polygons[channel][index]["polygon"] = list(zip(y_points, x_points))


    return polygons
def get_polygons_points(
    polygons: dict, image_size: tuple, ) -> dict:


    for channel in polygons.keys():
        for index, polygon in enumerate(polygons[channel]):
            x_points = [element[0][0] for element in polygon["polygon"]]
            y_points = [element[0][1] for element in polygon["polygon"]]
            x_points = [
                int(element) for element in x_points
            ]
            y_points = [
                int(element )  for element in y_points
            ]
            x_points = [
                int(element) if element < image_size else int(image_size)
                for element in x_points
            ]
            y_points = [
                int(element) if element < image_size else int(image_size)
                for element in y_points
            ]
            x_points = [int(element) if element > 0 else 0 for element in x_points]
            y_points = [int(element) if element > 0 else 0 for element in y_points]
            assert max(x_points) <= image_size
            assert min(x_points) >= 0
            assert max(y_points) <= image_size
            assert min(y_points) >= 0
            polygons[channel][index]["polygon"] = list(zip(x_points, y_points))


    return polygons
def get_predicted_polygons(
    probas: np.ndarray, min_cc: int, classes_names: list
) -> dict:

    page_contours = {}
    max_probas = np.argmax(probas, axis=0)
    for channel in range(0, probas.shape[0]-1):

        # 找到对应类别的位置，将true false矩阵转成 1 0矩阵 保留当前类别位置的概率值  提取出当前类别的概率图

        channel_probas = np.uint8(max_probas == channel) * probas[channel, :, :]
        # 二值化图像，当前类别为感兴趣的部分设为1
        bin_img = channel_probas.copy()
        bin_img[bin_img > 0] = 1
        # 使用 OpenCV 的 findContours 函数提取图像中的轮廓。cv2.RETR_EXTERNAL 只检索最外层的轮廓，cv2.CHAIN_APPROX_SIMPLE 用于压缩水平、垂直和对角线段的轮廓，只保留端点。
        # hierarchy存储的结构[Next, Previous, First Child, Parent]
        # Next: 同层的下一个轮廓的索引。
        # Previous: 同层的上一个轮廓的索引。
        # First Child: 第一个子轮廓的索引。
        # Parent: 父轮廓的索引。
        # contours 是一个包含多个轮廓的列表。每个轮廓是一个 numpy 数组，其中每个元素是一个轮廓点的坐标。这些坐标通常以 [[x, y]] 的形式存储，其中 x 和 y
        contours, hierarchy = cv2.findContours(
            np.uint8(bin_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        #过滤掉所有面积小于 min_cc 的轮廓。这样可以去除小的噪点。
        if min_cc > 0:
            contours = [
                contour for contour in contours if cv2.contourArea(contour) > min_cc
            ]
        page_contours[classes_names[channel]] = [
            {
                "confidence": compute_confidence(contour, channel_probas),
                "polygon": contour,
            }
            for contour in contours
        ]
    return page_contours
def adjust_polygons(polygons,position,inputsize):
    points=position.split('_')
    xs=int(points[0])
    ys=int(points[1])
    xe=int(points[2])
    ye=int(points[3])
    adjusted_polygons = {}
    height,width=inputsize[0],inputsize[1]
    for cls, poly_list in polygons.items():
        adjusted_polygons[cls] = []

        for poly_info in poly_list:
            confidence = poly_info['confidence']
            original_polygons = []

            adjusted_polygon = [
                (min(x + xs, width),  min(y + ys, height))
                for x, y in poly_info["polygon"]
            ]

            original_polygons.append({
                'confidence': confidence,
                'polygon': adjusted_polygon
            })

            adjusted_polygons[cls].extend(original_polygons)

    return adjusted_polygons



def compute_confidence(region: np.ndarray, probas: np.ndarray) -> float:

    mask = np.zeros(probas.shape)
    # 在region多边形区域画轮廓 [region] 是多边形的坐标点列表，0 表示绘制第一个轮廓，在这里指的就是唯一的轮廓，1 是绘制颜色，-1 表示填充整个轮廓区域。结果是，mask 中多边形区域的像素点会被标记为 1，其它区域保持为 0
    cv2.drawContours(mask, [region], 0, 1, -1)

    # 用多边形区域内像素概率值的总和除以像素总数，得到该区域的平均概率值，即置信度分数。
    confidence = np.sum(mask * probas) / np.sum(mask)
    return round(confidence, 4)


def json_serialize(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def save_prediction(polygons: dict, filename: str):

    base, ext = os.path.splitext(filename)
    with open(filename.replace(ext, ".json"), "w",encoding='utf-8') as outfile:
        json.dump(polygons, outfile, default=json_serialize, indent=4)


def save_prediction_image(polygons, colors, input_size, filename,filename_combined,original_image):

    image = np.zeros((input_size[0], input_size[1], 3),dtype=np.uint8)
    image_formask=image.copy()
    image_formask[:] = [128, 128, 128]
    index = 0
    for channel in polygons.keys():
        if channel == "img_size":
            continue
        color = [int(element) for element in colors[index]]
        for polygon in polygons[channel]:
            # 如果遇到有padding的部分超出了图像也不会报错，会自动画图像内的部分
            cv2.drawContours(image, [np.array(polygon["polygon"])], 0, color, -1)
            cv2.polylines(image, [np.array(polygon["polygon"])], True, color, thickness=5)
            cv2.drawContours(image_formask, [np.array(polygon["polygon"])], 0, color, -1)
        index += 1
    io.imsave(filename, np.uint8(image_formask))
    # 将原始图像与多边形图像融合
    blended_image = cv2.addWeighted(original_image["image"], 0.7, image, 0.3, 0)
    # 保存融合后的图像
    io.imsave(filename_combined, np.uint8(blended_image))





def save_prediction_image_combined(polygons, colors, input_size, filename,filename_combined,original_image,image,image_formask):
    if image is None:
        image = np.zeros((input_size[0], input_size[1], 3),dtype=np.uint8)
        image_formask = image.copy()
        image_formask[:] = [128, 128, 128]
    objet_parent=os.path.join(os.path.dirname(filename_combined),'objet_predit')
    if not os.path.exists(objet_parent):
        os.makedirs(objet_parent)
    index_objet=0
    index = 0
    for channel in polygons.keys():
        if channel == "img_size":
            continue
        color = [int(element) for element in colors[index]]
        for polygon in polygons[channel]:
            # 值为 0，表示我们只绘制列表中的第一个轮廓。如果列表中只有一个轮廓，那么使用 0 是正确的选择。
            # color:这是绘制轮廓的颜色。颜色通常是一个三元组，
            # -当值为 -1 时，表示填充轮廓内部区域。如果是正数，则表示轮廓线的厚度（单位是像素）。
            cv2.drawContours(image, [np.array(polygon["polygon"])], 0, color, -1)
            cv2.polylines(image, [np.array(polygon["polygon"])], True, color, thickness=5)
            cv2.drawContours(image_formask, [np.array(polygon["polygon"])], 0, color, -1)
            if channel =="math":
                polygon_points = np.array(polygon["polygon"], np.int32)
                # 获取多边形的边界框
                x, y, w, h = cv2.boundingRect(polygon_points)
                x_end=x+w
                y_end=y+h
                if x_end>original_image.shape[1] or y_end>original_image.shape[0]:
                    continue
                # 从原始图像中截取多边形所在的外接矩形区域
                cropped_polygon = original_image[y:y_end, x:x_end]
                # 创建一个与外接矩形区域大小一致的空白掩膜
                polygon_mask = np.zeros((h, w), dtype=np.uint8)
                polygon_mask2 = np.zeros((h, w,3), dtype=np.uint8)
                # 将多边形坐标偏移到相对于外接矩形的局部坐标
                polygon_shifted = polygon_points - [x, y]
                # 在掩膜中绘制多边形
                cv2.fillPoly(polygon_mask, [polygon_shifted], 255)
                # 仅保留掩膜中的多边形部分
                cropped_polygon_with_mask = np.zeros_like(cropped_polygon)

                cropped_polygon_with_mask[polygon_mask==255] = cropped_polygon[polygon_mask==255]

                cv2.fillPoly(polygon_mask2, [polygon_shifted], [255,0,0])
                blended_image_mask = cv2.addWeighted(cropped_polygon, 0.7, polygon_mask2, 0.3, 0)
                # 保存截取的多边形图像，只包含多边形区域
                name_img=os.path.basename(filename).split('.jpg')[0]
                cropped_filename = os.path.join(objet_parent,f"{name_img}_x_{x}_y_{y}_w_{w}_h_{h}.png")
                cropped_filename2 = os.path.join(objet_parent, f"{name_img}_contenu_x_{x}_y_{y}_w_{w}_h_{h}.png")
                io.imsave(cropped_filename, np.uint8(blended_image_mask))
                io.imsave(cropped_filename2, np.uint8(cropped_polygon_with_mask))


        index += 1
    io.imsave(filename, np.uint8(image_formask))
    # 将原始图像与多边形图像融合
    blended_image = cv2.addWeighted(original_image, 0.7, image, 0.3, 0)
    # 保存融合后的图像
    io.imsave(filename_combined, np.uint8(blended_image))
    return image,image_formask,True





