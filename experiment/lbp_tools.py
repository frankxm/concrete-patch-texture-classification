import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage.feature.texture import local_binary_pattern


def region_lbp(image, radii, ps, levels=256, standardize=False):
    # float(0,1)转unit8
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    # 找出所有值为0的位置，后面计算Lbp时排除掉
    mask = np.where(image == 0)
    if standardize:
        # 前景像素
        true_pixels_mask = np.where(image > 0)
        texture_mean = np.mean(image[true_pixels_mask])
        texture_std = np.std(image[true_pixels_mask])

        standardized_image = np.copy(image)
        # [u-3.1sigma,u+3.1sigma]限制范围，用来去除极端值Outlier，防止极端点影响lbp
        # 正态分布中u±sigma包含68%数据，u±2sigma包含95%，u±3sigma包含99.7%
        limits = [round(texture_mean - 3.1 * texture_std), round(texture_mean + 3.1 * texture_std)]
        standardized_image[np.where(standardized_image < limits[0])] = limits[0]
        standardized_image[np.where(standardized_image > limits[1])] = limits[1]
        # 量化，目的稳定lbp，小波动都压缩到一个bin，局部变化重要，绝对灰度不重要.步骤是先归一化到[0,1]再乘以level最后再离散化(floor)
        bin_image = np.floor(
            levels * (standardized_image.astype(float) - limits[0]) / (limits[1] - limits[0] + 1)).astype(np.uint16)
    else:
        bin_image = np.floor(levels * image.astype(float) / 256.0).astype(np.uint16)

    # plt.figure(figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Gray Image')
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(bin_image, cmap='gray')
    # plt.title(f'Binned Image(bins={levels})')
    # plt.axis('off')
    # plt.show()
    # 正常范围是[0,levels-1]，这里把背景(padding)设为非法值
    bin_image[mask] = levels


    # H,W,R,P
    lbps = np.zeros((image.shape[0], image.shape[1], len(radii), len(ps)))
    for i, radius in enumerate(radii):
        for j, p in enumerate(ps):
            # lbp值正常范围[0,2**p -1]
            lbp = local_binary_pattern(bin_image, p, radius, method='default')
            lbp[mask] = 2**p
            lbps[..., i, j] = lbp

    return lbps


def get_lbp_histograms(lbp, bins):

    hs = []
    for i in range(lbp.shape[3]):
        for j in range(lbp.shape[2]):
            current_image = lbp[..., j, i]
            # 获取LBP map（一个像素点对应一个值，一个图像则对应一个map）中对应原图像中像素值为0的位置（因为之前把它设为了最大）
            mask_value = np.max(current_image)
            # 把不同points维度下的lbpmap进行量化，比如8points时维度是256维，而16points时维度是65536维
            # 归一化到[0,1] -> 缩放到bins -> 最后再离散floor
            bin_image = np.floor(bins * current_image / mask_value)

            values, counts = np.unique(bin_image, return_counts=True)

            h = np.zeros((1, bins))
            for k in range(len(values)):
                # 过滤掉0像素值，为非法值bins，正常为[0,bins-1]
                if not values[k] == bins:
                    h[0, int(values[k])] = counts[k]
            h = h/np.sum(h)
            # 保存每个尺度下的lbp值的出现频率
            hs.append(h)

            # plt.figure()
            # plt.bar(range(bins), h.flatten(), color='skyblue')
            # plt.xlabel('Bin')
            # plt.ylabel('Normalized Count')
            # plt.title(f'LBP Histogram: radius_index={j}, points_index={i}')
            # plt.show()
    # 多个尺度下拼接再展平，[1,bins] -> [n,bins] -> [n*bins]
    features = np.ravel(np.concatenate(hs))[None, ...]
    return features


def get_lbp_feature_names(radii, ps, bins):
    lbp_feature_names = []
    for p in ps:
        for radius in radii:
            for bin in range(bins):
                lbp_feature_names.append("lbp_radius{:02d}_p{:02d}_bin{:03d}".format(radius, p, bin))
    return lbp_feature_names