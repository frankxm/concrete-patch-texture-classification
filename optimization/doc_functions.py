# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

    Generic functions used during all the steps.
"""

import copy
import math

import numpy as np
import pywt
import torch
import logging
import config
config_dict = config.__dict__
from preprocessing import get_texture

class Sampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data,
        batch_size,
        no_of_epochs,
        israndom,
        generator
    ):
        self.batch_size = batch_size
        self.num_epochs = no_of_epochs
        self.current_epoch = 0
        self.israndom = israndom
        self.generator = generator

        # 所有样本的 index
        self.real_indices = list(range(len(data)))

    def __len__(self):
        # 你的实际 __iter__ 会丢弃 size=1 的最后一个 batch，
        # 所以这里用近似值即可
        return math.ceil(len(self.real_indices) / self.batch_size)

    def __iter__(self):

        logging.info(
            f"real images in train:{len(self.real_indices)}"
        ) if self.israndom else logging.info(
            f"real images in valid:{len(self.real_indices)}"
        )


        # 1. 随机打乱所有 index

        shuffled_indices = torch.randperm(
            len(self.real_indices),
            generator=self.generator
        ).tolist()

        indices = [
            self.real_indices[i]
            for i in shuffled_indices
        ]


        # 2. 按 batch_size 切分

        final_indices = [
            indices[i:i + self.batch_size]
            for i in range(
                0,
                len(indices),
                self.batch_size
            )
        ]

        # 3. 只删除 batch size = 1 的情况≥2 的不完整 batch 保留

        final_indices = [
            batch
            for batch in final_indices
            if len(batch) > 1
        ]

        #  打乱 batch 的顺序
        batch_order = torch.randperm(
            len(final_indices),
            generator=self.generator
        ).tolist()

        final_indices = [
            final_indices[i]
            for i in batch_order
        ]

        self.current_epoch += 1

        return iter(final_indices)
def pad_images_masks(
    images, image_padding_value
):

    heights = [element.shape[0] for element in images]
    widths = [element.shape[1] for element in images]
    max_height = max(heights)
    max_width = max(widths)

    # Make the tensor shape be divisible by 8.
    if max_height % 8 != 0:
        max_height = int(8 * np.ceil(max_height / 8))
    if max_width % 8 != 0:
        max_width = int(8 * np.ceil(max_width / 8))
    # 创建一个批次，维度为batchsize  height width 3
    padded_images = (
        np.ones((len(images), max_height, max_width, images[0].shape[2]))
        * image_padding_value
    )

    for index, image in enumerate(images):
        delta_h = max_height - image.shape[0]
        delta_w = max_width - image.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_images[
            index,
            top : padded_images.shape[1] - bottom,
            left : padded_images.shape[2] - right,
            :,
        ] = image



    return padded_images




# DataLoader 将获取到的样本数据传递给 collate_fn 函数。collate_fn 函数定义了如何将这些样本数据组合成一个批次（batch）
class DLACollateFunction:
    def __init__(self,model_name=None,mean_features=None,std_features=None):
        self.image_padding_token = 0
        self.mask_padding_token = 4
        self.model_name=model_name
        self.mean_features=mean_features
        self.std_features=std_features



    def __call__(self, batch):
        image = [item["image"] for item in batch]
        mask = [item["label"] for item in batch]
        mask_extra=[item["label_extra"] for item in batch]
        pad_image=image
        if self.model_name=='texture_model' :
            X, names, feature_names =get_texture(image, **config_dict)
            X_scaled = (X - self.mean_features) / self.std_features
            # mean_check = X_scaled.mean(axis=0)
            # std_check = X_scaled.std(axis=0)
            #
            # print("每个特征的均值 (应该接近 0):", mean_check)
            # print("每个特征的标准差 (应该接近 1):", std_check)

            # X_unsqueezed=torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)
            X_unsqueezed = torch.from_numpy(
                X_scaled.astype(np.float32, copy=False)
            ).unsqueeze(0).unsqueeze(0)

            if self.model_name=='texture_model':
                return {
                    "image": X_unsqueezed.permute(2, 0, 3, 1),
                    "label":  torch.as_tensor(mask, dtype=torch.long),
                    "label_extra": torch.as_tensor(mask_extra, dtype=torch.long),
                }

        image_array = np.stack(pad_image, axis=0).astype(
            np.float32,
            copy=False
        )
        return {
            "image": torch.from_numpy(image_array).permute(0, 3, 1, 2),
            "label":torch.as_tensor(mask, dtype=torch.long),
            "label_extra": torch.as_tensor(mask_extra, dtype=torch.long),
        }


class DLACollateFunction_for_prediction:
    def __init__(self,model_name=None,mean=None,std=None):
        self.model_name=model_name
        self.mean_features=mean
        self.std_features=std
    def __call__(self, batch):


        if self.model_name in ['latefusionmodel','midfusionmodel']:
            image = [item["image_original"] for item in batch]
            name = [item["name"] for item in batch]
            image_normalized = [item["image"] for item in batch]
            X, names, feature_names = get_texture(image, **config_dict)
            X_scaled = (X - self.mean_features) / self.std_features
            # X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)
            #
            # return {
            # "texture": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
            # "name":name,
            # "image": torch.tensor(image_normalized).permute(0, 3, 1, 2),
            # }
            X_unsqueezed = torch.from_numpy(
                X_scaled.astype(np.float32, copy=False)
            ).unsqueeze(0).unsqueeze(0)

            image_array = np.stack(image_normalized, axis=0)

            return {
                "texture": X_unsqueezed.permute(2, 0, 3, 1),
                "name": name,
                "image": torch.from_numpy(image_array).permute(0, 3, 1, 2),
            }
        else:
            image = [item["image"] for item in batch]
            name = [item["name"] for item in batch]
            X, names, feature_names = get_texture(image, **config_dict)
            X_scaled = (X - self.mean_features) / self.std_features
            # X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)
            # return {
            #     "image": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
            #     "name": name,
            # }
            texture = torch.from_numpy(
                X_scaled.astype(np.float32, copy=False)
            ).unsqueeze(0).unsqueeze(0).permute(2, 0, 3, 1)

            return {
                "image": texture,
                "name": name,
            }


# DataLoader 将获取到的样本数据传递给 collate_fn 函数。collate_fn 函数定义了如何将这些样本数据组合成一个批次（batch）
class DLACollateFunction_multimodal:
    def __init__(self,model_name=None,mean_features=None,std_features=None):
        self.image_padding_token = 0
        self.mask_padding_token = 4
        self.model_name=model_name
        self.mean_features=mean_features
        self.std_features=std_features



    def __call__(self, batch):
        image = [item["image_original"] for item in batch]
        mask = [item["label"] for item in batch]
        mask_extra = [item["label_extra"] for item in batch]
        image_normalized = [item["image"] for item in batch]
        X, names, feature_names = get_texture(image, **config_dict)
        X_scaled = (X - self.mean_features) / self.std_features
        # X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)
        texture = torch.from_numpy(
            X_scaled.astype(np.float32, copy=False)
        ).unsqueeze(0).unsqueeze(0).permute(2, 0, 3, 1)
        image_array = np.stack(image_normalized, axis=0).astype(
            np.float32,
            copy=False
        )
        image_tensor = torch.from_numpy(
            image_array
        ).permute(0, 3, 1, 2)
        label_tensor = torch.as_tensor(mask, dtype=torch.long)
        label_extra_tensor = torch.as_tensor(
            mask_extra,
            dtype=torch.long
        )
        return {
            "texture": texture,
            "label": label_tensor,
            "label_extra": label_extra_tensor,
            "image": image_tensor,
        }
        # return {
        #     "texture": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
        #     "label": torch.tensor(mask),
        #     "label_extra": torch.tensor(mask_extra),
        #     "image": torch.tensor(image_normalized).permute(0, 3, 1, 2),
        # }

def compute_wavelet_features(image, wavelet='db1', level=5):

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    # # haar小波单级变换之后：
    # # 低频信息的取值范围为：[0,510]
    # # 高频信息的取值范围为：[-255,255]
    # coeffs = pywt.dwt2(image, 'haar')
    # cA, (cH, cV, cD) = coeffs
    # # cA_uint8=cA.astype(np.uint8)
    # cH_uint8 = cH.astype(np.uint8)
    # cV_uint8 = cV.astype(np.uint8)
    # cD_uint8 = cD.astype(np.uint8)
    # cA_uint8 = np.clip(cA, 0, 255)
    # # cH_uint8 = np.clip(cH, 0, 255)
    # # cV_uint8 = np.clip(cV, 0, 255)
    # # cD_uint8 = np.clip(cD, 0, 255)
    # # 将各个子图进行拼接，最后得到一张图
    # AH = np.concatenate([cA_uint8, cH_uint8], axis=1)
    # VD = np.concatenate([cV_uint8, cD_uint8], axis=1)
    # img = np.concatenate([AH, VD], axis=0)
    # # 显示灰度图
    # plt.subplot(1,2,1)
    # plt.imshow(image, cmap='gray')
    # plt.title('original image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img, cmap='gray')
    # plt.title('2d-wavelet 1 level')
    #
    # # 二级变换
    # coeffs = pywt.wavedec2(image, 'haar', level=2)
    # cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    #
    # # 将每个子图的像素范围都归一化到与CA2一致  CA2 [0,255* 2**level]
    # AH2 = np.concatenate([cA2, cH2 + 510], axis=1)
    # VD2 = np.concatenate([cV2 + 510, cD2 + 510], axis=1)
    # cA1 = np.concatenate([AH2, VD2], axis=0)
    #
    # AH = np.concatenate([cA1, (cH1 + 255) * 2], axis=1)
    # VD = np.concatenate([(cV1 + 255) * 2, (cD1 + 255) * 2], axis=1)
    # img = np.concatenate([AH, VD], axis=0)
    # plt.figure(2)
    # plt.imshow(img.astype(np.uint8), 'gray')
    # plt.title('2D WT')
    # plt.show()


    features = []
    for coeff_level in coeffs[1:]:  # Skip approximation
        for coeff in coeff_level:  # Horizontal, Vertical, Diagonal
            features.append(np.mean(coeff))
            features.append(np.std(coeff))

    return np.array(features)

def compute_fractal_dimension(image, threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    Z = image < threshold * image.max()
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, int(size)) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return np.array([coeffs[0]])  # fractal dimension
