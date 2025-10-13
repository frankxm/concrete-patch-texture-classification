# -*- coding: utf-8 -*-

"""
    The predict module
    ======================

    Use it to predict some images from a trained network.
"""

import logging
import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import prediction as pr_utils
import json
import pandas as pd

def predict_multimodal(
    prediction_path: str,
    log_path: str,
    classes_names: list,
    loaders: dict,
    net,
    img_dir,
    use_gpu,
):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Run prediction.
    net.eval()

    logging.info("Starting predicting")
    starting_time = time.time()
    num_img=0
    results=[]
    class_map = {str(ind):cls for ind,cls in enumerate(classes_names)}
    output_imgdir = os.path.join(log_path, prediction_path, 'test','pred_images')
    os.makedirs(output_imgdir, exist_ok=True)
    with torch.no_grad():
        for index, (set, loader) in enumerate(zip(["test"], loaders.values())):
            for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):

                logging.info(f"inference of image:{data['name'][0]}")
                num_img += 1

                output = net(data["image"].to(device).float(),data["texture"].float(),'prediction')
                conf_top2, preds_top2 = torch.topk(output, k=2, dim=1)
                top1_class = preds_top2[0, 0].item()
                top1_conf = round(conf_top2[0, 0].item(), 4)
                top2_class = preds_top2[0, 1].item()
                top2_conf = round(conf_top2[0, 1].item(), 4)
                results.append({
                    "image": data['name'][0],
                    "class": top1_class,
                    "class_confidence": top1_conf,
                    "class2": top2_class,
                    "class2_confidence": top2_conf
                })

                img_current = img_dir[data['name'][0]]
                text = f"Pred: {class_map[str(top1_class)]}"
                text2 = f"Conf: {top1_conf}"
                text3 = f"Top2: {class_map[str(top2_class)]}"
                text4 = f"Conf2: {top2_conf}"

                position = (10, 30)
                position2 = (10, 50)
                position3 = (10, 70)
                position4 = (10, 90)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 0, 255)
                thickness = 2

                cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text3, position3, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)

                # 保存带文字的图片
                output_path = os.path.join(output_imgdir, f"{data['name'][0]}.png")
                cv2.imwrite(output_path, img_current)
                # conf, preds = torch.max(output, 1)
                # results.append({
                #     "image": data['name'][0],
                #     "class": preds.item(),
                #     "confidence":round(conf.item(),4)
                # })
                # img_current=img_dir[data['name'][0]]
                # # 设置文本参数
                # text = f"Pred: {class_map[str(preds.item())]}"
                # text2=f"Confidence:{round(conf.item(),4)}"
                # position = (10, 30)  # 左上角稍下的位置
                # position2 = (10, 50)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.5
                # color = (0, 0, 255)
                # thickness = 2
                # # 写到图像上
                # cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                # cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                # output_path=os.path.join(output_imgdir,f"{data['name'][0]}.png")
                # cv2.imwrite(output_path, img_current)


    df = pd.DataFrame(results)
    csv_output_path =os.path.join(log_path, prediction_path, set,"predictions.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to: {csv_output_path}")

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )



def run(
    prediction_path: str,
    log_path: str,
    classes_names: list,
    loaders: dict,
    net,
    img_dir,
    use_gpu,
):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Run prediction.
    net.eval()

    logging.info("Starting predicting")
    starting_time = time.time()
    num_img=0
    results=[]
    class_map = {str(ind):cls for ind,cls in enumerate(classes_names)}
    output_imgdir = os.path.join(log_path, prediction_path, 'test','pred_images')
    os.makedirs(output_imgdir, exist_ok=True)

    with torch.no_grad():
        for index, (set, loader) in enumerate(zip(["test"], loaders.values())):


            for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):

                logging.info(f"inference of image:{data['name'][0]}")
                num_img += 1
                output = net(data["image"].to(device).float(),'prediction')
                conf_top2, preds_top2 = torch.topk(output, k=2, dim=1)
                top1_class = preds_top2[0, 0].item()
                top1_conf = round(conf_top2[0, 0].item(), 4)
                top2_class = preds_top2[0, 1].item()
                top2_conf = round(conf_top2[0, 1].item(), 4)
                results.append({
                    "image": data['name'][0],
                    "class": top1_class,
                    "class_confidence": top1_conf,
                    "class2": top2_class,
                    "class2_confidence": top2_conf
                })

                img_current = img_dir[data['name'][0]]
                text = f"Pred: {class_map[str(top1_class)]}"
                text2 = f"Conf: {top1_conf}"
                text3 = f"Top2: {class_map[str(top2_class)]}"
                text4 = f"Conf2: {top2_conf}"

                position = (10, 30)
                position2 = (10, 50)
                position3 = (10, 70)
                position4 = (10, 90)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 0, 255)
                thickness = 2

                cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text3, position3, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img_current, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)

                # 保存带文字的图片
                output_path = os.path.join(output_imgdir, f"{data['name'][0]}.png")
                cv2.imwrite(output_path, img_current)
                # conf, preds = torch.max(output, 1)
                # results.append({
                #     "image": data['name'][0],
                #     "class": preds.item(),
                #     "confidence":round(conf.item(),4)
                # })
                # img_current=img_dir[data['name'][0]]
                # # 设置文本参数
                # text = f"Pred: {class_map[str(preds.item())]}"
                # text2=f"Confidence:{round(conf.item(),4)}"
                # position = (10, 30)  # 左上角稍下的位置
                # position2 = (10, 50)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.5
                # color = (0, 0, 255)
                # thickness = 2
                # # 写到图像上
                # cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                # cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                # output_path=os.path.join(output_imgdir,f"{data['name'][0]}.png")
                # cv2.imwrite(output_path, img_current)


    df = pd.DataFrame(results)
    csv_output_path =os.path.join(log_path, prediction_path, set,"predictions.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to: {csv_output_path}")

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )



