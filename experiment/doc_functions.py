# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

    Generic functions used during all the steps.
"""

import copy
import os

import random


import pywt
import torch
import logging



import config
config_dict = config.__dict__
from lbp_tools import *
from glcm_tools import *

def rgb_to_gray_value(rgb: tuple) -> int:
    """
    Compute the gray value of a RGB tuple.
    :param rgb: The RGB value to transform.
    :return: The corresponding gray value.
    """
    try:
        return int(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    except TypeError:
        return int(int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114)


def rgb_to_gray_array(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the gray array (NxM) of a RGB array (NxMx3).
    :param rgb: The RGB array to transform.
    :return: The corresponding gray array.
    """
    gray_array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.uint8(gray_array)


def create_buckets(images_sizes, bin_size):
    """
    Group images into same size buckets.
    :param images_sizes: The sizes of the images.
    :param bin_size: The step between two buckets.
    :return bucket: The images indices grouped by size.
    """

    max_size = max([image_size for image_size in images_sizes.values()])
    min_size = min([image_size for image_size in images_sizes.values()])
    # binsize为每个桶的尺寸范围，先创建空桶，每个桶的最大尺寸作为键
    bucket = {}
    current = min_size + bin_size - 1
    while current < max_size:
        bucket[current] = []
        current += bin_size
    bucket[max_size] = []
    # 遍历图像尺寸分配到特定桶
    for index, value in images_sizes.items():
        # 计算当前尺寸所属的桶的区域，计算上限
        dict_index = (((value - min_size) // bin_size) + 1) * bin_size + min_size - 1
        bucket[min(dict_index, max_size)].append(index)
    # 删除空桶，只保留有图像的桶
    bucket = {
        dict_index: values for dict_index, values in bucket.items() if len(values) > 0
    }
    return bucket

class Sampler(torch.utils.data.Sampler):
    def __init__(self, data, bin_size, batch_size,no_of_epochs,israndom,generator):
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.data_sizes = [(sample[0].shape[0],sample[0].shape[1]) for sample in data.values()]
        self.num_epochs = no_of_epochs
        self.current_epoch = 0
        self.israndom=israndom


        self.real_indices = [i for i, sample in enumerate(data) ]

        # 区分水平和竖直图像
        self.vertical = {
            index: sample[1]
            for index, sample in enumerate(self.data_sizes)
            if sample[0] > sample[1]
        }
        self.horizontal = {
            index: sample[0]
            for index, sample in enumerate(self.data_sizes)
            if sample[0] <= sample[1]
        }
        # 创建竖直图像桶和水平图像桶
        self.buckets = [
            create_buckets(self.vertical, self.bin_size)
            if len(self.vertical) > 0
            else {},
            create_buckets(self.horizontal, self.bin_size)
            if len(self.horizontal) > 0
            else {},
        ]

        self.generator=generator

    def __len__(self):
        total_batches = math.ceil(len(self.real_indices) / self.batch_size)
        return total_batches

    def __iter__(self):
        print(f"{self.current_epoch}")
        buckets = copy.deepcopy(self.buckets)
        # 打乱每种桶中每个键的图像索引
        for index, bucket in enumerate(buckets):
            for key in bucket.keys():
                lst = bucket[key]
                indices = torch.randperm(len(lst), generator=self.generator).tolist()
                bucket[key] = [lst[i] for i in indices]
                # random.shuffle(buckets[index][key])

        mixed_indices = self.real_indices
        indices = torch.randperm(len(mixed_indices), generator=self.generator).tolist()
        mixed_indices = [mixed_indices[i] for i in indices]
        # random.shuffle(mixed_indices)
        logging.info(f"real images in train:{len(self.real_indices)} ") if self.israndom else logging.info(f"real images in valid:{len(self.real_indices)} ")


        # 按批次分组，根据每个桶的每个键逆序遍历，依次加入到final_indices数组中。每当达到batchsize时，批次增加索引增加。最后在打乱所有批次。
        if self.batch_size is not None:
            final_indices = []
            index_current = -1
            for bucket in buckets:
                current_batch_size = self.batch_size
                for key in sorted(bucket.keys(), reverse=True):
                    for index in bucket[key]:
                        if index in mixed_indices:
                            if current_batch_size + 1 > self.batch_size:
                                current_batch_size = 0
                                final_indices.append([])
                                index_current += 1
                            current_batch_size += 1
                            final_indices[index_current].append(index)
            # random.shuffle(final_indices)
            indices = torch.randperm(len(final_indices), generator=self.generator).tolist()
            final_indices = [final_indices[i] for i in indices]

        self.current_epoch+=1
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
        pad_image=image
        if self.model_name=='texture_model' :
            X, names, feature_names =get_texture(image, **config_dict)
            X_scaled = (X - self.mean_features) / self.std_features
            # mean_check = X_scaled.mean(axis=0)
            # std_check = X_scaled.std(axis=0)
            #
            # print("每个特征的均值 (应该接近 0):", mean_check)
            # print("每个特征的标准差 (应该接近 1):", std_check)

            X_unsqueezed=torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)


            if self.model_name=='texture_model':
                return {
                    "image": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
                    "label": torch.tensor(mask),
                }


        return {
            "image": torch.tensor(pad_image).permute(0, 3, 1, 2),
            "label": torch.tensor(mask),
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
            X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)

            return {
            "texture": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
            "name":name,
            "image": torch.tensor(image_normalized).permute(0, 3, 1, 2),
            }
        else:
            image = [item["image"] for item in batch]
            name = [item["name"] for item in batch]
            X, names, feature_names = get_texture(image, **config_dict)
            X_scaled = (X - self.mean_features) / self.std_features
            X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)
            return {
                "image": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
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
        image_normalized = [item["image"] for item in batch]
        X, names, feature_names = get_texture(image, **config_dict)
        X_scaled = (X - self.mean_features) / self.std_features
        X_unsqueezed = torch.tensor(X_scaled).unsqueeze(0).unsqueeze(0)

        return {
            "texture": torch.tensor(X_unsqueezed).permute(2, 0, 3, 1),
            "label": torch.tensor(mask),
            "image": torch.tensor(image_normalized).permute(0, 3, 1, 2),
        }

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

def get_texture(batch_images,**kwargs):

    # GLCM parameters
    distances = kwargs["distances"]
    angles = kwargs["angles"]
    standardize_glcm_image = kwargs["standardize_glcm_image"]
    glcm_levels = kwargs["glcm_levels"]
    props = kwargs["props"]

    # LBP parameters
    ps = kwargs["ps"]
    radii = kwargs["radii"]
    standardize_lbp_image = kwargs["standardize_lbp_image"]
    lbp_levels = kwargs["lbp_levels"]
    bins = kwargs["bins"]

    names = []
    X = []
    for img in batch_images:
        img_gray = rgb_to_gray_array(img)
        # bin_image是将原始灰度图 image 做过处理（标准化/离散化）后的版本。它是一个整数类型图像，其像素值范围在 [0, levels-1] 之间，用于作为 graycomatrix 的输入。如果启用了 standardize=True，它是经过均值±3.1倍标准差截断 + 归一化之后的版本。
        # 图像中 image == 0 的区域会被视为“掩膜”区域，被填为 levels，然后在计算 GLCM 时排除。
        glcms, bin_image = region_glcm(img_gray, distances, angles, glcm_levels, standardize=standardize_glcm_image)

        glcm_features = get_glcm_features(glcms, props)

        lbps = region_lbp(img_gray, radii, ps, lbp_levels, standardize=standardize_lbp_image)
        lbps_features = get_lbp_histograms(lbps, bins)

        # Wavelet
        wavelet_feats = compute_wavelet_features(img_gray)
        # Fractal
        fractal_feat = compute_fractal_dimension(img_gray)

        all_features = np.concatenate([
            glcm_features[0, :],
            lbps_features[0, :],
            wavelet_feats,
            fractal_feat
        ])
        all_features = np.concatenate([
            glcm_features[0, :],
            lbps_features[0, :],
        ])
        X.append(all_features)
        # X.append(np.concatenate((glcm_features[0, :], lbps_features[0, :])))

    glcm_feature_names = get_glcm_feature_names(distances, angles, props)
    lbps_feature_names = get_lbp_feature_names(radii, ps, bins)
    feature_names = glcm_feature_names + lbps_feature_names
    names = np.array(names)
    X = np.array(X)



    return X, names, np.array(feature_names)[None, ...]