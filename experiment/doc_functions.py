# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

    Generic functions used during all the steps.
"""

import copy
import pywt
import torch
import logging
import config
config_dict = config.__dict__
from lbp_tools import *
from glcm_tools import *
from preprocessing import get_texture

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
#
# class Sampler(torch.utils.data.Sampler):
#     def __init__(self, data, batch_size,no_of_epochs,israndom,generator):
#         self.batch_size = batch_size
#         self.data_sizes = [(sample[0].shape[0],sample[0].shape[1]) for sample in data.values()]
#         self.num_epochs = no_of_epochs
#         self.current_epoch = 0
#         self.israndom=israndom
#         self.bin_size = 20
#
#
#         self.real_indices = [i for i, sample in enumerate(data) ]
#
#         # 区分水平和竖直图像
#         self.vertical = {
#             index: sample[1]
#             for index, sample in enumerate(self.data_sizes)
#             if sample[0] > sample[1]
#         }
#         self.horizontal = {
#             index: sample[0]
#             for index, sample in enumerate(self.data_sizes)
#             if sample[0] <= sample[1]
#         }
#         # 创建竖直图像桶和水平图像桶
#         self.buckets = [
#             create_buckets(self.vertical, self.bin_size)
#             if len(self.vertical) > 0
#             else {},
#             create_buckets(self.horizontal, self.bin_size)
#             if len(self.horizontal) > 0
#             else {},
#         ]
#
#         self.generator=generator
#
#     def __len__(self):
#         total_batches = math.ceil(len(self.real_indices) / self.batch_size)
#         return total_batches
#
#     def __iter__(self):
#         print(f"{self.current_epoch}")
#         buckets = copy.deepcopy(self.buckets)
#         # 打乱每种桶中每个键的图像索引
#         for index, bucket in enumerate(buckets):
#             for key in bucket.keys():
#                 lst = bucket[key]
#                 indices = torch.randperm(len(lst), generator=self.generator).tolist()
#                 bucket[key] = [lst[i] for i in indices]
#                 # random.shuffle(buckets[index][key])
#
#         mixed_indices = self.real_indices
#         indices = torch.randperm(len(mixed_indices), generator=self.generator).tolist()
#         mixed_indices = [mixed_indices[i] for i in indices]
#         # random.shuffle(mixed_indices)
#         logging.info(f"real images in train:{len(self.real_indices)} ") if self.israndom else logging.info(f"real images in valid:{len(self.real_indices)} ")
#
#
#         # 按批次分组，根据每个桶的每个键逆序遍历，依次加入到final_indices数组中。每当达到batchsize时，批次增加索引增加。最后在打乱所有批次。
#         if self.batch_size is not None:
#             final_indices = []
#             index_current = -1
#             for bucket in buckets:
#                 current_batch_size = self.batch_size
#                 for key in sorted(bucket.keys(), reverse=True):
#                     for index in bucket[key]:
#                         if index in mixed_indices:
#                             if current_batch_size + 1 > self.batch_size:
#                                 current_batch_size = 0
#                                 final_indices.append([])
#                                 index_current += 1
#                             current_batch_size += 1
#                             final_indices[index_current].append(index)
#             # 如果某个 batch 只有 1 个样本，则丢弃  batch >= 2 的全部保留，即使小于 batch_size
#             final_indices = [
#                 batch for batch in final_indices
#                 if len(batch) > 1
#             ]
#
#             # random.shuffle(final_indices)
#             indices = torch.randperm(len(final_indices), generator=self.generator).tolist()
#             final_indices = [final_indices[i] for i in indices]
#
#
#         self.current_epoch+=1
#         return iter(final_indices)

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
            X =get_texture(image, **config_dict)
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
            X= get_texture(image, **config_dict)
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
            X = get_texture(image, **config_dict)
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
        X= get_texture(image, **config_dict)
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

