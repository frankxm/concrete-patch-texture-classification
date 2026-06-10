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

import pandas as pd
import math
from models.efficientformer import Attention,Meta3D,Meta4D
from gradcam_utils import GradCAM, show_cam_on_image
from vit_rollout import VITAttentionRollout


# checkpoints ensemble
def predict_multimodal(
    prediction_path: str,
    log_path: str,
    classes_names: list,
    loaders: dict,
    nets,
    img_dir,
    use_gpu,
    weights
):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    for net in nets:
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

                # output = net(data["image"].to(device).float(),data["texture"].to(device).float(),'prediction')

                logits_list = []
                for net in nets:
                    out = net(
                        data["image"].to(device).float(),
                        data["texture"].to(device).float(),
                        'train'
                    )
                    logits_list.append(out)

                logits_stack = torch.stack(logits_list, dim=0)

                # ensemble moyenne
                output = logits_stack.mean(dim=0)

                # ensemble medianne
                # output = torch.median(logits_stack, dim=0).values


                #weights sum
                # weights_t = torch.tensor(weights, device=logits_stack.device, dtype=logits_stack.dtype)
                # weights_t = weights_t / weights_t.sum()
                # output = (logits_stack * weights_t[:, None, None]).sum(dim=0)


                # weights medium
                # weights_t = torch.tensor(weights, device=logits_stack.device, dtype=logits_stack.dtype)
                # weights_t = weights_t[:, None, None]
                # weighted_logits = logits_stack * weights_t
                # output = torch.median(weighted_logits, dim=0).values

                prob = torch.softmax(output, dim=1)

                conf_top2, preds_top2 = torch.topk(prob, k=2, dim=1)
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
    nets:list,
    img_dir,
    use_gpu,
    visualization,
    mean,std,
    weights
):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    for net in nets:
        net.eval()

    logging.info("Starting predicting")
    starting_time = time.time()
    num_img=0
    results=[]
    class_map = {str(ind):cls for ind,cls in enumerate(classes_names)}
    output_imgdir = os.path.join(log_path, prediction_path, 'test','pred_images')
    os.makedirs(output_imgdir, exist_ok=True)
    # gradcam 不同stage的特征热图可视化，表示模型在做这个预测时，依赖了图像的哪些区域 descriminative regions，哪些区域对当前类别判别的Logits贡献较大,哪些位置的feature被认为重要
    # cnn feature 的判别区域 虽然是单层可视化，但本身当前层的激活值是通过前面层前向传播得到的，因此决策从低层纹理到局部到语义到高层语义，gradcam的每一层热图是隐式包含前层信息的，编码了前面的信息，做最终决策可用最后层的图。


    if visualization:
        # attention_rollout(net,loaders,device,mean,std)
        gradcam_visualization(net,loaders,device,mean,std)

    with torch.no_grad():
        for index, (set, loader) in enumerate(zip(["test"], loaders.values())):


            for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):

                logging.info(f"inference of image:{data['name'][0]}")
                num_img += 1
                # output = net(data["image"].to(device).float(),'prediction')
                logits_list = []

                for net in nets:
                    out= net(data["image"].to(device).float(),'train')
                    logits_list.append(out)

                logits_stack = torch.stack(logits_list, dim=0)

                # ensemble moyenne
                output = logits_stack.mean(dim=0)

                # ensemble medianne
                # output = torch.median(logits_stack, dim=0).values

                # weights sum
                # weights_t = torch.tensor(weights, device=logits_stack.device, dtype=logits_stack.dtype)
                # weights_t = weights_t / weights_t.sum()
                # output = (logits_stack * weights_t[:, None, None]).sum(dim=0)

                # weights medium
                # weights_t = torch.tensor(weights, device=logits_stack.device, dtype=logits_stack.dtype)
                # weights_t = weights_t[:, None, None]
                # weighted_logits = logits_stack * weights_t
                # output = torch.median(weighted_logits, dim=0).values

                prob = torch.softmax(output, dim=1)

                # 可视化最后一层meta3d的注意力矩阵，多头平均融合
                # if visualization:
                #     attn_list = []
                #     for m in net.modules():
                #         if isinstance(m, Attention) and hasattr(m, "attn_visual"):
                #             attn_list.append(m.attn_visual)
                #     # vit_num(4)个atten_list,4个meta3d  [4,1,8,49,49]
                #     assert len(attn_list) > 0
                #     last_attn = attn_list[-1][0]  # [heads=8, N, N]
                #     attn_mean = last_attn.mean(0)  # [N, N]
                #
                #     print("Attention map size:", attn_mean.shape)
                #     # token数为49，atten_map为49*49
                #     N = attn_mean.shape[0]
                #     h = w = int(math.sqrt(N))
                #     assert h * w == N
                #     # 固定列j（key)把所有行i的值相加：第j个patch被多少别的patch的总关注，不区分类别的注意力，只能表示哪些patch被关注的比较多，无法知道哪些patch对哪些类别贡献大
                #     # dim=0沿着行方向相加
                #     attn_patch = attn_mean.sum(dim=0)
                #     attn_2d = attn_patch.reshape(h, w).cpu().numpy()
                #     # 缩放为原大小会有误差
                #     attn_up = cv2.resize(attn_2d, (224, 224),
                #                          interpolation=cv2.INTER_NEAREST)
                #     # min-max标准化为[0,1]，适合可视化
                #     attn_up = (attn_up - attn_up.min()) / (attn_up.max() + 1e-6)
                #
                #     image_cpu = data["image"][0][0].detach().cpu()
                #     mean = np.array(mean[0]).reshape( 1, 1)
                #     std = np.array(std[0]).reshape( 1, 1)
                #     image_denorm = image_cpu * std + mean
                #     image_denorm = image_denorm.numpy()
                #     image_to_save = np.clip(image_denorm, 0, 255)
                #     # 接近原始图像，可视化
                #     image_original = image_to_save.astype(np.uint8)
                #     # 未作反归一化，不真实
                #     img = data["image"][0].permute(1, 2, 0).cpu().numpy()
                #
                #     plt.figure(figsize=(10, 4))
                #
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(image_original, cmap="gray")
                #     plt.title("Original gray image")
                #     plt.axis("off")
                #
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(attn_up, cmap="jet")
                #     plt.colorbar(fraction=0.046, pad=0.04)
                #     plt.title("Total Attention Received (Patch Map)")
                #     plt.axis("off")
                #
                #     plt.subplot(1,3, 3)
                #     plt.imshow(img, cmap="gray")
                #     plt.imshow(attn_up, cmap="jet", alpha=0.5)
                #     plt.colorbar(fraction=0.046, pad=0.04)
                #     plt.title("Overlay on Input Image")
                #     plt.axis("off")
                #
                #
                #
                #     plt.tight_layout()
                #     plt.show()



                conf_top2, preds_top2 = torch.topk(prob, k=2, dim=1)
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


#
# def run(
#     prediction_path: str,
#     log_path: str,
#     classes_names: list,
#     loaders: dict,
#     net,
#     img_dir,
#     use_gpu,
#     visualization,
#     mean,std
# ):
#     if use_gpu:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device("cpu")
#
#     # Run prediction.
#     net.eval()
#
#     logging.info("Starting predicting")
#     starting_time = time.time()
#     num_img=0
#     results=[]
#     class_map = {str(ind):cls for ind,cls in enumerate(classes_names)}
#     output_imgdir = os.path.join(log_path, prediction_path, 'test','pred_images')
#     os.makedirs(output_imgdir, exist_ok=True)
#     # gradcam 不同stage的特征热图可视化，表示模型在做这个预测时，依赖了图像的哪些区域 descriminative regions，哪些区域对当前类别判别的Logits贡献较大,哪些位置的feature被认为重要
#     # cnn feature 的判别区域 虽然是单层可视化，但本身当前层的激活值是通过前面层前向传播得到的，因此决策从低层纹理到局部到语义到高层语义，gradcam的每一层热图是隐式包含前层信息的，编码了前面的信息，做最终决策可用最后层的图。
#
#
#     if visualization:
#         # attention_rollout(net,loaders,device,mean,std)
#         gradcam_visualization(net,loaders,device,mean,std)
#
#     with torch.no_grad():
#         for index, (set, loader) in enumerate(zip(["test"], loaders.values())):
#
#
#             for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):
#
#                 logging.info(f"inference of image:{data['name'][0]}")
#                 num_img += 1
#                 output = net(data["image"].to(device).float(),'prediction')
#                 # 可视化最后一层meta3d的注意力矩阵，多头平均融合
#                 # if visualization:
#                 #     attn_list = []
#                 #     for m in net.modules():
#                 #         if isinstance(m, Attention) and hasattr(m, "attn_visual"):
#                 #             attn_list.append(m.attn_visual)
#                 #     # vit_num(4)个atten_list,4个meta3d  [4,1,8,49,49]
#                 #     assert len(attn_list) > 0
#                 #     last_attn = attn_list[-1][0]  # [heads=8, N, N]
#                 #     attn_mean = last_attn.mean(0)  # [N, N]
#                 #
#                 #     print("Attention map size:", attn_mean.shape)
#                 #     # token数为49，atten_map为49*49
#                 #     N = attn_mean.shape[0]
#                 #     h = w = int(math.sqrt(N))
#                 #     assert h * w == N
#                 #     # 固定列j（key)把所有行i的值相加：第j个patch被多少别的patch的总关注，不区分类别的注意力，只能表示哪些patch被关注的比较多，无法知道哪些patch对哪些类别贡献大
#                 #     # dim=0沿着行方向相加
#                 #     attn_patch = attn_mean.sum(dim=0)
#                 #     attn_2d = attn_patch.reshape(h, w).cpu().numpy()
#                 #     # 缩放为原大小会有误差
#                 #     attn_up = cv2.resize(attn_2d, (224, 224),
#                 #                          interpolation=cv2.INTER_NEAREST)
#                 #     # min-max标准化为[0,1]，适合可视化
#                 #     attn_up = (attn_up - attn_up.min()) / (attn_up.max() + 1e-6)
#                 #
#                 #     image_cpu = data["image"][0][0].detach().cpu()
#                 #     mean = np.array(mean[0]).reshape( 1, 1)
#                 #     std = np.array(std[0]).reshape( 1, 1)
#                 #     image_denorm = image_cpu * std + mean
#                 #     image_denorm = image_denorm.numpy()
#                 #     image_to_save = np.clip(image_denorm, 0, 255)
#                 #     # 接近原始图像，可视化
#                 #     image_original = image_to_save.astype(np.uint8)
#                 #     # 未作反归一化，不真实
#                 #     img = data["image"][0].permute(1, 2, 0).cpu().numpy()
#                 #
#                 #     plt.figure(figsize=(10, 4))
#                 #
#                 #     plt.subplot(1, 3, 1)
#                 #     plt.imshow(image_original, cmap="gray")
#                 #     plt.title("Original gray image")
#                 #     plt.axis("off")
#                 #
#                 #     plt.subplot(1, 3, 2)
#                 #     plt.imshow(attn_up, cmap="jet")
#                 #     plt.colorbar(fraction=0.046, pad=0.04)
#                 #     plt.title("Total Attention Received (Patch Map)")
#                 #     plt.axis("off")
#                 #
#                 #     plt.subplot(1,3, 3)
#                 #     plt.imshow(img, cmap="gray")
#                 #     plt.imshow(attn_up, cmap="jet", alpha=0.5)
#                 #     plt.colorbar(fraction=0.046, pad=0.04)
#                 #     plt.title("Overlay on Input Image")
#                 #     plt.axis("off")
#                 #
#                 #
#                 #
#                 #     plt.tight_layout()
#                 #     plt.show()
#
#
#
#                 conf_top2, preds_top2 = torch.topk(output, k=2, dim=1)
#                 top1_class = preds_top2[0, 0].item()
#                 top1_conf = round(conf_top2[0, 0].item(), 4)
#                 top2_class = preds_top2[0, 1].item()
#                 top2_conf = round(conf_top2[0, 1].item(), 4)
#                 results.append({
#                     "image": data['name'][0],
#                     "class": top1_class,
#                     "class_confidence": top1_conf,
#                     "class2": top2_class,
#                     "class2_confidence": top2_conf
#                 })
#
#                 img_current = img_dir[data['name'][0]]
#                 text = f"Pred: {class_map[str(top1_class)]}"
#                 text2 = f"Conf: {top1_conf}"
#                 text3 = f"Top2: {class_map[str(top2_class)]}"
#                 text4 = f"Conf2: {top2_conf}"
#
#                 position = (10, 30)
#                 position2 = (10, 50)
#                 position3 = (10, 70)
#                 position4 = (10, 90)
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5
#                 color = (0, 0, 255)
#                 thickness = 2
#
#                 cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
#                 cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
#                 cv2.putText(img_current, text3, position3, font, font_scale, color, thickness, cv2.LINE_AA)
#                 cv2.putText(img_current, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)
#
#                 # 保存带文字的图片
#                 output_path = os.path.join(output_imgdir, f"{data['name'][0]}.png")
#                 cv2.imwrite(output_path, img_current)
#                 # conf, preds = torch.max(output, 1)
#                 # results.append({
#                 #     "image": data['name'][0],
#                 #     "class": preds.item(),
#                 #     "confidence":round(conf.item(),4)
#                 # })
#                 # img_current=img_dir[data['name'][0]]
#                 # # 设置文本参数
#                 # text = f"Pred: {class_map[str(preds.item())]}"
#                 # text2=f"Confidence:{round(conf.item(),4)}"
#                 # position = (10, 30)  # 左上角稍下的位置
#                 # position2 = (10, 50)
#                 # font = cv2.FONT_HERSHEY_SIMPLEX
#                 # font_scale = 0.5
#                 # color = (0, 0, 255)
#                 # thickness = 2
#                 # # 写到图像上
#                 # cv2.putText(img_current, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
#                 # cv2.putText(img_current, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
#                 # output_path=os.path.join(output_imgdir,f"{data['name'][0]}.png")
#                 # cv2.imwrite(output_path, img_current)
#
#
#     df = pd.DataFrame(results)
#     csv_output_path =os.path.join(log_path, prediction_path, set,"predictions.csv")
#     df.to_csv(csv_output_path, index=False)
#     print(f"Predictions saved to: {csv_output_path}")
#
#     end = time.gmtime(time.time() - starting_time)
#     logging.info(
#         "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
#     )

# 获取特征图热力图
def gradcam_visualization(net,loaders,device,mean,std):
    for index, (set, loader) in enumerate(zip(["test"], loaders.values())):
        for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):
            # eval模式保证推理结果稳定，但需要删除with torch.no_grad(),梯度计算图需回传
            net.eval()
            output = net(data["image"].to(device).float(), 'prediction')
            target_category = output.argmax(dim=1).item()
            # gt_category = data["label"].item()
            prob = output[0, target_category].item()
            # is_correct = (target_category == gt_category)


            # efficientformer 的不同stage中分别有4,4,12,2的meta4d
            stages_meta4d = []
            for module in net.network:
                if isinstance(module, torch.nn.Sequential):
                    meta4d_list = [m for m in module if isinstance(m, Meta4D)]
                    if len(meta4d_list) > 0:
                        stages_meta4d.append(meta4d_list)

            stages_meta3d = []
            for module in net.network:
                if isinstance(module, torch.nn.Sequential):
                    meta3d_list = [m for m in module if isinstance(m, Meta3D)]
                    if len(meta3d_list) > 0:
                        stages_meta3d.append(meta3d_list)

            # 第三层的meta4d太多有12个，选5个展示
            stage3_blocks = stages_meta4d[2]
            sample_indices = [0,3,6,9,11]
            stages_meta4d[2] = [stage3_blocks[i] for i in sample_indices]

            stages_all=stages_meta4d+stages_meta3d

            num_stages = len(stages_all)
            max_blocks = max(len(s) for s in stages_all)

            # 反归一化图像
            image_cpu = data["image"][0][0].detach().cpu()
            mean_arr = np.array(mean[0]).reshape(1, 1)
            std_arr = np.array(std[0]).reshape(1, 1)
            image_denorm = image_cpu * std_arr + mean_arr
            image_denorm = image_denorm.numpy()
            image_to_save = np.clip(image_denorm, 0, 255)
            image_original = image_to_save.astype(np.uint8)
            image_original = np.repeat(image_original[:, :, None], 3, axis=2)

            fig = plt.figure(figsize=(4 * max_blocks, 4 * num_stages))
            title_color = "green"
            class_names = ["fluid", "good", "dry", "tearing", "ecrase"]
            fig.suptitle(f"{data['name'][0]} stage evolution heatmap|"+
                f"Prediction: {class_names[target_category]} ({prob:.2f}) ",
                # f"GT: {class_names[gt_category]}   |   "
                # f"Correct: {is_correct}",
                fontsize=22
                # color=title_color
            )





            plot_index = 1

            for stage_idx, meta_blocks in enumerate(stages_all):
                for block_idx, block in enumerate(meta_blocks):
                    # ViT 是只有 cls token 参与分类，剩下的14*14=196个patch token 梯度为 0 因此如果使用vit，对最后一层取gradcam，会全蓝
                    if stage_idx==4:
                        cam = GradCAM(
                            model=net,
                            target_layers=[block],
                            use_cuda=False if device == "cpu" else True,
                            reshape_transform=EfficientFormerReshapeTransform()
                        )
                    else:
                        cam = GradCAM(
                            model=net,
                            target_layers=[block],
                            use_cuda=False if device == "cpu" else True,
                        )

                    grayscale_cam = cam(
                        input_tensor = data["image"].to(device).float(),
                        target_category=target_category
                    )[0]

                    _, _, overlay = show_cam_on_image(
                        image_original / 255.,
                        grayscale_cam,
                        use_rgb=True
                    )

                    plt.subplot(num_stages, max_blocks, plot_index)
                    plt.imshow(overlay)
                    plt.title(f"Stage {stage_idx + 1}\nBlock {block_idx + 1}")
                    plt.axis("off")

                    plot_index += 1


                if block_idx + 1 < max_blocks:
                    plt.subplot(num_stages, max_blocks, plot_index)
                    plt.imshow(image_original)
                    plt.title(f"Original image")
                    plt.axis("off")


            save_dir = "./test"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{data['name'][0]}_stage_evolution.png"
            )

            plt.subplots_adjust(
                hspace=0.4,  # 行间距
                wspace=0.25,  # 列间距
                top=0.92  # 给总标题留空间
            )
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()


# 针对attention block，获取网络的attention信息流，强调token交互，哪些输出token被输入token关注
# 显式累积前层的 attention：每层 attention 矩阵都乘以前面所有层的矩阵 → 最终 heatmap是全局累计
def attention_rollout(net,loaders,device,mean,std):
    for index, (set, loader) in enumerate(zip(["test"], loaders.values())):
        for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):
            net.eval()
            output = net(data["image"].to(device).float(), 'prediction')
            target_category = output.argmax(dim=1).item()


            # 反归一化图像
            image_cpu = data["image"][0][0].detach().cpu()
            mean_arr = np.array(mean[0]).reshape(1, 1)
            std_arr = np.array(std[0]).reshape(1, 1)
            image_denorm = image_cpu * std_arr + mean_arr
            image_denorm = image_denorm.numpy()
            image_to_save = np.clip(image_denorm, 0, 255)
            image_original = image_to_save.astype(np.uint8)
            image_original = np.repeat(image_original[:, :, None], 3, axis=2)

            attn_rollout = VITAttentionRollout(net, discard_ratio=0.9, head_fusion='mean',target_category=target_category,use_grad=False,use_clstoken=False,device=device )
            masks_per_layer = attn_rollout(data["image"].to(device).float())
            prob = output[0, target_category].item()



            num_layers = len(masks_per_layer)

            fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))
            class_names = ["fluid", "good", "dry", "tearing", "ecrase"]

            for i, mask in enumerate(masks_per_layer):
                # resize到原图大小 默认情况双线性插值interpolation=cv2.INTER_LINEAR，平滑渐变热力图。interpolation=cv2.INTER_NEAREST这个会变成一块块
                mask_resized = cv2.resize(mask, (image_original.shape[1], image_original.shape[0]))

                # 生成 heatmap
                img_float = np.float32(image_original) / 255

                heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
                # opencv默认是bgr类型的heatmap，需要转成rgb来展示
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                # 归一化到[0,1]
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + img_float
                cam = cam / np.max(cam)
                cam_uint8 = np.uint8(255 * cam)



                axes[0,i].imshow(cam_uint8)
                axes[0,i].set_title(f"Attention layer {i + 1}")
                axes[0,i].axis('off')


                # 这里Imshow mask_resized+cmap和直接Imshow heatmap差不多一样
                im = axes[1, i].imshow(mask_resized, cmap='jet')
                axes[1, i].set_title(f"Layer {i + 1} Heatmap")
                axes[1, i].axis('off')

                fig.colorbar(im,ax=axes[1, i],fraction=0.046, pad=0.04)



            # 最后一张显示原图
            axes[0,-1].imshow(image_original)
            axes[0,-1].set_title("Original Image")
            axes[0,-1].axis('off')

            # 第二行最后一格设为空
            axes[1, -1].axis('off')
            axes[1, -1].set_visible(False)

            fig.suptitle(f"{data['name'][0]} Attention Rollout | "
                         f"Prediction: {class_names[target_category]} ({prob:.2f})",
                         fontsize=18)
            plt.tight_layout()
            save_dir = "./test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{data['name'][0]}_attentionrollout_layers_mean.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)






class EfficientFormerReshapeTransform:
    def __init__(self, h=7, w=7):
        self.h = h
        self.w = w

    def __call__(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        x = x.reshape(B, self.h, self.w, C)
        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]
        return x
