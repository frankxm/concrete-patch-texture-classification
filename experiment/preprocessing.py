# -*- coding: utf-8 -*-

"""
    The preprocessing module
    ======================

    Use it to preprocess the images.
"""

import os


import cv2


import torch
from torch.utils.data import Dataset

from doc_functions import rgb_to_gray_array, rgb_to_gray_value
import random
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pandas as pd
import visualize_features
import config
config_dict = config.__dict__
from lbp_tools import *
from glcm_tools import *
from skimage.util import img_as_ubyte
import pywt

class TrainingDataset(Dataset):

    def __init__(
            self, augment_all: list, transform: list = None,augmentations_transformation: list = None, augmentations_pixel: list = None,forbid=False,model_name='',generator=None
    ):
        self.images = [sample[0] for sample in augment_all.values()]
        self.labels = [sample[1] for sample in augment_all.values()]
        self.transform = transform
        self.augmentations_transformation = augmentations_transformation if augmentations_transformation else []
        self.augmentations_pixel = augmentations_pixel if augmentations_pixel else []
        self.forbid=forbid
        self.model_name=model_name
        self.generator=generator

    def __len__(self) -> int:

        return len(self.images)

    def randint(self, low, high):
        return int(torch.randint(low, high, (1,), generator=self.generator).item())

    def rand(self):
        return float(torch.rand(1, generator=self.generator).item())

    def choice(self, seq):
        idx = self.randint(0, len(seq))
        return seq[idx]

    def __getitem__(self, idx: int) -> dict:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]
        sample = {"image": image, "label": label,"size": image.shape[0:2]}


        if not self.forbid and self.augmentations_transformation and self.augmentations_pixel:

            # 增强的概率为0.8  增强情况下某一类为0.8*0.5*0.25
            if self.rand() < 0.8:
                # 两类增强方式概率0.5
                if self.rand() < 0.5:
                    aug = self.choice(self.augmentations_transformation)
                    image = aug(image,self.generator)
                    logging.info(f'operation {aug.__name__} for current image,label')
                else:
                    aug = self.choice(self.augmentations_pixel)
                    image = aug(image,self.generator)
                    logging.info(f'operation {aug.__name__} for current image')
                # plt.close()
                # plt.figure(figsize=(15, 8))
                # plt.subplot(3,2,1)
                # plt.imshow(sample["image"])
                # plt.subplot(3,2,2)
                # plt.imshow(gray_array_to_rgb(sample["mask"]))
                # plt.subplot(3,2,3)
                # plt.imshow(image)
                # plt.subplot(3,2,4)
                # plt.imshow(gray_array_to_rgb(label))
                # plt.subplot(3, 2, 5)
                # plt.imshow(sample["mask_binary"], cmap='gray')
                # plt.subplot(3, 2, 6)
                # plt.imshow(label_binary,cmap='gray')
                # plt.savefig(f'./Augmentation/{aug.__name__}_{idx}.png')

                sample["image"] = image
                sample["label"] = label

                # all_images = [image for image in self.images]
                # mean, std = compute_mean_std(all_images, batch_size=100)  # 使用分批次计算均值和标准差
                # means.append(mean)
                # stds.append(std)

            else:
                logging.info(f'no operation for current image')

        if self.transform and self.model_name != "texture_model":
            if self.model_name=='midfusionmodel':
                sample ["image_original"]=sample["image"]
            sample = self.transform(sample)


        return sample




class PredictionDataset(Dataset):
    def __init__(self, data, transform: list = None,model_name=''):

        self.images = [sample for sample in data.values()]
        self.imgnames = list(data.keys())
        self.transform = transform
        self.model_name=model_name
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]
        imgname=self.imgnames[idx]



        sample = {
            "image": img,
            "name": imgname,
            "size": img.shape[0:2],
        }
        if self.model_name != 'texture_model':
            if self.model_name in ['latefusionmodel','midfusionmodel']:
                sample ["image_original"]=img
            if self.transform:
                sample = self.transform(sample)
        return sample




class Rescale:
    """
    The Rescale class is used to rescale the image of a sample into a
    given size.
    """

    def __init__(self, output_size: int):
        """
        Constructor of the Rescale class.
        :param output_size: The desired new size.
        """
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        """
        Rescale the sample image into the model input size.
        :param sample: The sample to rescale.
        :return sample: The rescaled sample.
        """
        old_size = sample["image"].shape[:2]
        # Compute the new sizes.
        ratio = float(self.output_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]

        # Resize the image.
        if max(old_size) != self.output_size:
            image = cv2.resize(sample["image"], (new_size[1], new_size[0]))
            sample["image"] = image

        # Resize the label. MUST BE AVOIDED.
        if "mask" in sample.keys():
            if max(sample["mask"].shape[:2]) != self.output_size:
                mask = cv2.resize(sample["mask"], (new_size[1], new_size[0]))
                sample["mask"] = mask
        return sample


class Pad:
    """
    The Pad class is used to pad the image of a sample to make it divisible by 8.
    保持图像的大致宽高比并将其尺寸调整为8的倍数，通过适当的padding来实现，这是为了确保在处理过程中不会显著改变图像的原始比例，同时又能满足计算要求
    """

    def __init__(self):
        """
        Constructor of the Pad class.
        """
        pass

    def __call__(self, sample: dict) -> dict:
        """
        Pad the sample image with zeros.
        :param sample: The sample to pad.
        :return sample: The padded sample.
        """
        # Compute the padding parameters.
        delta_w = 0
        delta_h = 0
        if sample["image"].shape[0] % 8 != 0:
            delta_h = (
                    int(8 * np.ceil(sample["image"].shape[0] / 8))
                    - sample["image"].shape[0]
            )
        if sample["image"].shape[1] % 8 != 0:
            delta_w = (
                    int(8 * np.ceil(sample["image"].shape[1] / 8))
                    - sample["image"].shape[1]
            )

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Add padding to have same size images.
        image = cv2.copyMakeBorder(
            sample["image"],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        sample["image"] = image
        sample["padding"] = {"top": top, "left": left,"bottom":bottom,"right":right}
        return sample


class Normalize:
    """
    The Normalize class is used to normalize the image of a sample.
    The mean value and standard deviation must be first computed on the
    training dataset.
    """

    def __init__(self, mean: list, std: list):
        """
        Constructor of the Normalize class.
        :param mean: The mean values (one for each channel) of the images
                     pixels of the training dataset.
        :param std: The standard deviations (one for each channel) of the
                    images pixels of the training dataset.
        """
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        # 在归一化前不能先设定image类型为uin8，因为在计算过程中用的是浮点计算，而最后又被转成uint8，这样就会使得数值会被截取在0-255之间。从而影响后续逆归一化图像的显示
        image = np.zeros(sample["image"].shape,dtype=np.float64)
        for channel in range(sample["image"].shape[2]):
            image[:, :, channel] = (
                                           np.float64(sample["image"][:, :, channel]) - self.mean[channel]
                                   ) / self.std[channel]

        sample["image"] = image
        return sample


class ToTensor:
    """
    The ToTensor class is used convert ndarrays into Tensors.
    """

    def __call__(self, sample: dict) -> dict:
        """
        Transform the sample image and label into Tensors.
        :param sample: The initial sample.
        :return sample: The sample made of Tensors.
        """
        sample["image"] = torch.from_numpy(sample["image"].transpose((2, 0, 1)))
        if "mask" in sample.keys():
            sample["mask"] = torch.from_numpy(sample["mask"])
        return sample


# 将图像的四个顶点随机移动，从而改变图像的透视效果
def random_perspective_transform(image,generator: torch.Generator):
    height, width = image.shape[:2]
    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    # 在大小的正百分之十和负百分之十之间变换
    max_shift = int(min(height, width) * 0.1)
    # delta = random.randint(-max_shift, max_shift)
    delta = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
    dst_points = np.float32([
        [delta, delta],
        [width - 1 - delta, delta],
        [width - 1 - delta, height - 1 - delta],
        [delta, height - 1 - delta]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image, matrix, (width, height))

    return image


# 对图像进行随机弹性变形，使图像看起来像被拉伸或挤压

def random_elastic_transform(image, generator,alpha=6, sigma=4):
    # random_state = np.random.RandomState(None)
    # shape = image.shape[:2]  # 只取图像的高度和宽度
    # dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    shape = image.shape[:2]  # H x W

    # 使用 torch 生成随机数并转换为 numpy
    rand_x = torch.rand(shape, generator=generator).numpy() * 2 - 1
    rand_y = torch.rand(shape, generator=generator).numpy() * 2 - 1

    dx = gaussian_filter(rand_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(rand_y, sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 对每个通道分别进行弹性变形
    transformed_image = np.zeros_like(image)
    for c in range(image.shape[2]):
        transformed_image[..., c] = map_coordinates(image[..., c], indices, order=1, mode='reflect').reshape(shape)


    return transformed_image



def random_uniform(generator,low, high ):
    return float((high - low) * torch.rand(1, generator=generator) + low)

def random_randint(generator,low, high):
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


# 随机旋转图像和掩码，使它们倾斜一定的角度。
def random_rotate(image,generator):
    # angle = random.uniform(-180, 180)
    angle = random_uniform(generator, -180, 180)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return image

# 随机水平竖直混合翻转图像和掩码
def random_flip(image,generator):
    # -1混合，0竖直，1水平
    # flip_type = random.choice([-1, 0, 1])
    flip_type = int(random_uniform(generator, -1.5, 1.5))
    image = cv2.flip(image, flip_type)
    return image


# 平移图像和掩码
def random_shift(image,generator, max_shift=200):
    rows, cols = image.shape[:2]
    # tx = random.randint(-max_shift, max_shift)
    # ty = random.randint(-max_shift, max_shift)
    tx = random_randint(generator, -max_shift, max_shift)
    ty = random_randint(generator, -max_shift, max_shift)
    shift_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, shift_matrix, (cols, rows))

    return image



# 对图像应用高斯模糊，使图像变得模糊
# 高斯模糊只是针对图像的局部细节进行模糊化，不会显著改变全局的亮度或对比度，因此不会明显影响均值和方差。
def random_gaussian_blur(image,generator):
    radius = random_uniform(generator, 0.5, 1.2)
    # radius=random.uniform(0.5, 1.2)
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(image)


# 向图像添加高斯噪声，使图像看起来更嘈杂。
# 当 var 较大时，图像可能会变得过于嘈杂，从而改变其整体的均值和方差
def random_gaussian_noise(image,generator):
    row, col, ch = image.shape
    mean = 0
    var = random_uniform(generator, 1, 3)
    # var = random.uniform(1, 3)
    sigma = var ** 0.5
    # gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = torch.normal(mean, sigma, size=(row, col, ch), generator=generator).numpy()
    gauss = gauss.reshape(row, col, ch)
    image = image + gauss
    return np.clip(image, 0, 255).astype(np.uint8)


# 增强图像的锐度，使图像中的边缘变得更加清晰
# 锐化主要会增强边缘细节，但不会显著改变图像的全局亮度、对比度等特性。因此，对图像均值和方差的影响较小。
def random_sharpen(image,generator):
    factor = random_uniform(generator, 1.0, 2.0)
    # factor=random.uniform(1.0, 2.0)
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor)
    return np.array(image)

# 图像对比度增强  对比度调整的范围缩小至 0.8 - 1.2，这样增强操作不会对数据的方差产生剧烈变化。
def random_contrast(image,generator):
    factor = random_uniform(generator, 0.8, 1.2)
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    return np.array(image)


def gray_array_to_rgb(mask):
    color_map = {
        76: [255, 0, 0],  # Blue
        149: [0, 255, 0],  # Green
        29: [0, 0, 255],  # Red
        225: [255, 255, 0],  # Cyan
        105: [255, 0, 255],  # Magenta
        178: [0, 255, 255],  # Yellow
        127: [128, 128, 128]  # Gray
    }
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for gray_value, rgb_color in color_map.items():
        rgb_image[mask == gray_value] = rgb_color

    return rgb_image






def readimagelabel(image_folder,label_path,label_no_need):
    image_label_dict={}

    df = pd.read_csv(label_path)

    # dataset2 不考虑没有class2的ecrase情况
    # # 先筛掉 class=4 且 class2 为空的行
    # filtered_df = df[(df["class"] != 4) | ((df["class"] == 4) & df["class2"].notna())]
    # # 直接把 class==4 的行，替换成 class2 的值
    # filtered_df.loc[filtered_df["class"] == 4, "class"] = filtered_df["class2"]
    # df=filtered_df

    labels = df["class"].values
    image_names=df['image'].values


    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg','.tiff')):
            image_path = os.path.join(image_folder, filename)
            # 此时读取的是灰度图像，但是因为png能保存灰度格式图，读取时cv2自动复制了三遍灰度值到每个通道，视觉上和原图看起来一样，但实际上 shape 已经变了。
            image = cv2.imread(image_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]
            if image is None:
                continue
            base_name, _ = os.path.splitext(filename)
            image_names_list = image_names.tolist()
            if filename not in image_names_list:
                continue
            if not label_no_need:
                label = labels[image_names_list.index(filename)]
                image_label_dict[base_name]=(image,label)
            # # 预测阶段label没用，训练阶段有用，不需要获取label，所以label随意设置，后续也用不到
            else:
                label=''
                image_label_dict[base_name] = (image,label)
    return image_label_dict



def compute_mean_std(image_list, batch_size=100):
    num_images = len(image_list)
    means = []
    stds = []

    for i in range(0, num_images, batch_size):
        batch = image_list[i:i + batch_size]
        batch_images = np.stack([np.array(img, dtype=np.float32) for img in batch])

        mean = np.mean(batch_images, axis=(0, 1, 2))
        std = np.std(batch_images, axis=(0, 1, 2))

        means.append(mean)
        stds.append(std)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


def resize_with_padding(image, target_size=299):
    h, w = image.shape[:2]

    # 计算缩放比例
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算填充大小
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1,2,2)
    # plt.imshow(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    # plt.show()

    return padded



def augmente_images(processed_image,name,label,image_label_dict):
    rotation_list=[-4, -3, -2, 2,  3, 4 ]
    # rotation
    for angle in rotation_list:
        name_current=f'{name}_angle{str(angle)}'
        image_center = tuple(np.array(processed_image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image_rotation = cv2.warpAffine(processed_image, rot_mat, processed_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        image_label_dict[name_current] = (image_rotation, label)
        # cv2.imwrite(f'{name_current}.png',image_rotation)

    # illumination rescale+shift
    alpha_beta_pairs = [(0.9, 0), (1.1, 0), (1.0, -50), (1.0, 50)]
    for pair in alpha_beta_pairs:
        name_current = f'{name}_a{str(pair[0])}_b{str(pair[1])}'
        alpha, beta =pair[0],pair[1]
        image_illumination = processed_image.astype(np.float32) * alpha + beta
        image_illumination=np.clip(image_illumination, 0, 255).astype(np.uint8)
        image_label_dict[name_current] = (image_illumination, label)
        # cv2.imwrite(f'{name_current}.png', image_illumination)

        #
        # plt.figure(alpha_beta_pairs.index(pair))
        # plt.imshow(image_illumination)
        # plt.show()



def apply_augmentations_and_compute_stats(imagedir, output_size, set,label_path,use_images_generees=False,images_generees_path=None,classes_names=None,num_genere=None):
    image_label_dict = readimagelabel(imagedir, label_path,label_no_need=False)

    processed_images=[]

    imagenamelist = [os.path.splitext(i)[0] for i in os.listdir(str(imagedir))]

    scale_list=[0.97, 0.98, 0.99,1.01, 1.02, 1.03]
    # 一张图共16种变化 207*17=3519
    for i, name in enumerate(tqdm(imagenamelist, desc=f"Augmente et Calcule mean std des images {set}")):
        try:
            sample = image_label_dict[name]
        except:
            # 出现不需要的图像就不加载，比如ecrase有时候不用
            continue
        image = sample[0]
        label = sample[1]
        # 原图padding
        processed_image = resize_with_padding(image, output_size)
        processed_images.append(processed_image)
        image_label_dict[name] = (processed_image, label)


        if set=='train':
            # 多种增强
            augmente_images(processed_image, name, label, image_label_dict)
            # 原图的不同scale
            for scale in scale_list:
                new_h = int(image.shape[0] * scale)
                new_w = int(image.shape[1] * scale)
                scaled_image = cv2.resize(image, (new_w, new_h) )
                processed_image = resize_with_padding(scaled_image,output_size)
                name_current = f'{name}_scale{str(scale)}'
                image_label_dict[name_current] = (processed_image, label)
                # cv2.imwrite(f'{name_current}.png',processed_image)

    if set == 'train' and use_images_generees:
        for cls in classes_names:
            cls_path=os.path.join(images_generees_path,cls.lower())
            num_samples=int(num_genere[classes_names.index(cls)])
            image_genere_namelist = [os.path.splitext(i)[0] for i in os.listdir(str(cls_path))]
            image_genere_label_dict = readimagelabel(cls_path, label_path, label_no_need=False)
            selected_names = random.sample(image_genere_namelist, num_samples)
            for i, name in enumerate(tqdm(selected_names, desc=f"Load images {cls} generees par stylegan3 {set}")):
                sample = image_genere_label_dict[name]
                image = sample[0]
                label = sample[1]
                # 原图padding
                processed_image = resize_with_padding(image, output_size)
                image_label_dict[name] = (processed_image, label)

                if set == 'train':
                    # 多种增强
                    augmente_images(processed_image, name, label, image_label_dict)
                    # 原图的不同scale
                    for scale in scale_list:
                        new_h = int(image.shape[0] * scale)
                        new_w = int(image.shape[1] * scale)
                        scaled_image = cv2.resize(image, (new_w, new_h))
                        processed_image = resize_with_padding(scaled_image, output_size)
                        name_current = f'{name}_scale{str(scale)}'
                        image_label_dict[name_current] = (processed_image, label)

    # 输出最终类别比例
    labels_list = [v[1] for v in image_label_dict.values()]
    label_arr = np.array(labels_list)
    unique_vals, counts = np.unique(label_arr, return_counts=True)
    ratios = counts / counts.sum()
    for val, cnt, ratio in zip(unique_vals, counts, ratios):
        print(f"值 {val}: 数量={cnt}, 比例={ratio:.2f}")

    proportion = {val: (cnt, ratio) for val, cnt, ratio in zip(unique_vals, counts, ratios)}


    if set =='train':
        mean_final, std_final = compute_mean_std(processed_images)

        logging.info(f"Mean in {set}: {mean_final}")
        logging.info(f"Std in {set}: {std_final}")




        # visualize_features.tsne(X,np.array(mask) )
        # visualize_features.pca(X, np.array(mask))

        return image_label_dict, mean_final, std_final,proportion
    else:
        return image_label_dict,proportion


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

def compute_sift_features(image, max_features=100):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_as_ubyte(image), None)
    if des is None:
        return np.zeros((max_features * 128,))
    if len(des) > max_features:
        des = des[:max_features]
    else:
        des = np.pad(des, ((0, max_features - len(des)), (0, 0)), 'constant')
    return des.flatten()
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
        img_gray=rgb_to_gray_array(img)
        glcms, bin_image = region_glcm(img_gray, distances, angles, glcm_levels, standardize=standardize_glcm_image)
        # 查看量化以及标准化后的bin_image
        # plt.figure(figsize=(5, 5))
        # plt.subplot(1,2,1)
        # plt.imshow(img_gray, cmap='gray')
        # plt.title('Gray Image')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(bin_image, cmap='gray')
        # plt.title('Binned Image(bins=12)')
        # plt.axis('off')
        # plt.show()
        # 查看glcm矩阵
        distance_index = 0  # 选择第几个距离
        angle_index = 0  # 选择第几个角度

        # plt.figure(figsize=(6, 6))
        # plt.imshow(glcms[:, :, distance_index, angle_index], cmap='hot')
        # plt.title(f'GLCM Matrix (distance={distances[distance_index]}, angle={angles[angle_index] * 180 / np.pi:.0f}°)')
        # plt.colorbar()
        # plt.xlabel('Gray level j')
        # plt.ylabel('Gray level i')
        # plt.show()
        glcm_features = get_glcm_features(glcms, props)

        lbps = region_lbp(img_gray, radii, ps, lbp_levels, standardize=standardize_lbp_image)
        lbps_features = get_lbp_histograms(lbps, bins)

        # for i in range(4):
        #     rad=radii[i]
        #     p=ps[0]
        #     lbp_img = lbps[..., i, 0]
        #     plt.figure(figsize=(6, 6))
        #     plt.subplot(1,2,1)
        #     plt.imshow(img_gray, cmap='gray')
        #     plt.title('Gray Image')
        #     plt.axis('off')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(lbp_img, cmap='gray')
        #     plt.title(f'LBP (radius={rad}, p={p})')
        #     plt.axis('off')
        #     plt.show()


        # Wavelet
        # wavelet_feats = compute_wavelet_features(img_gray)
        # Fractal
        # fractal_feat = compute_fractal_dimension(img_gray)
        # SIFT
        # sift_feat = compute_sift_features(img_gray)

        # all_features = np.concatenate([
        #     glcm_features[0, :],
        #     lbps_features[0, :],
        #     wavelet_feats,
        #     fractal_feat
        # ])
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
def apply_augmentations_and_compute_stats_pred(imagedir, output_size, set,label_path):
    image_label_dict = readimagelabel(imagedir, label_path,label_no_need=True)

    processed_images=[]

    imagenamelist = [os.path.splitext(i)[0] for i in os.listdir(str(imagedir))]

    for i, name in enumerate(tqdm(imagenamelist, desc=f"Resize and Calcule mean std des images {set}")):
        sample = image_label_dict[name]
        image = sample[0]
        processed_image = resize_with_padding(image,output_size)
        processed_images.append(processed_image)
        image_label_dict[name] = processed_image

    return image_label_dict

def show_augmented_images(image, augmentation_functions):

    for i, aug in enumerate(augmentation_functions):

        augname = str(aug.__name__)

        augmented_image = aug(image)
        plt.figure(i+1)
        plt.imshow(augmented_image,cmap='gray')
        plt.title(f'Augmented Image {augname}')
        plt.savefig(f'./augmentation_batch/{augname}.png')


if __name__ == '__main__':
    image_gray = cv2.imread(
        r"C:\Users\86139\Downloads\Baseline_modele\data\train\img_050-001.tiff")

    augmentations = [
        random_perspective_transform, random_elastic_transform, random_rotate, random_flip,random_gaussian_blur,random_gaussian_noise,random_sharpen,random_contrast
    ]


    show_augmented_images(image_gray, augmentations)

