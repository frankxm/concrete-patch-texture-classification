import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.feature.texture import local_binary_pattern



def region_glcm_lbp(image,mask,low,high, levels, type=None,glcm_param_list=None,lbp_param_list=None):
    if type=='lbp':
        standardize=lbp_param_list[2]
        levels=lbp_param_list[3]
    elif type=='glcm':
        standardize = glcm_param_list[2]
        levels = glcm_param_list[3]
    if standardize:
        bin_image = np.floor(
            levels * (image.astype(np.float32) - low)
            / (high - low + 1)
        ).astype(np.uint16)
    else:
        bin_image = (
                levels * image.astype(np.float32) / 256.0
        ).astype(np.uint16)
    # 查看量化以及标准化后的bin_image
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

    bin_image[mask] = levels

    if type=='glcm':
        distances=glcm_param_list[0]
        angles=glcm_param_list[1]
        props=glcm_param_list[4]
        glcms = graycomatrix(bin_image, distances, angles, levels + 1, symmetric=True)
        # 不取边界值，即原图的无效区域（像素为0）大小为(levels+1,levels+1,num_distances, num_angles)
        # Crop the row and column corresponding to intensity levels + 1
        glcms = glcms[:-1, :-1, ...].astype(np.float32)
        normalization = glcms.sum(axis=(0, 1), keepdims=True)
        glcms = np.divide(
            glcms,
            normalization,
            out=np.zeros_like(glcms),
            where=normalization != 0
        )
        features = np.empty(
            glcms.shape[2] * glcms.shape[3] * len(props),
            dtype=np.float32
        )
        n_da = glcms.shape[2] * glcms.shape[3]

        for idx, prop in enumerate(props):
            f = graycoprops(glcms, prop)

            features[
                idx * n_da:
                (idx + 1) * n_da
            ] = f.flatten('F')

        return features[None, :]

    elif type=='lbp':
        ps = lbp_param_list[0]
        radii = lbp_param_list[1]
        bins = lbp_param_list[4]
        features = []
        for p in ps:

            for radius in radii:
                # ① 计算 LBP
                lbp = local_binary_pattern(
                    bin_image,
                    p,
                    radius,
                    method='default'
                )

                # ② 把 padding/background 标记为非法值
                lbp[mask] = 2 ** p

                # ③ 只保留有效像素
                valid = lbp[lbp < 2 ** p]

                # ④ quantization
                bin_lbp = np.floor(
                    bins * valid / (2 ** p)
                ).astype(np.int32)

                # ⑤ histogram
                h = np.bincount(
                    bin_lbp,
                    minlength=bins
                ).astype(np.float32)

                # ⑥ normalization
                h /= h.sum()

                # ⑦ 保存的只是 histogram，不再保存完整 LBP map
                features.append(h)

        return np.concatenate(features)[None, :]



def get_glcm_features(glcm, props, avg_and_range=False):

    if avg_and_range:
        # len(distances) * len(angles) * len(props))
        features = np.zeros(glcm.shape[2] * 2 * len(props))

        for idx, prop in enumerate(props):
            # shape = (d, θ)
            f = graycoprops(glcm, prop)
            a = np.mean(f, -1)
            r = np.ptp(f, -1)
            features[idx*glcm.shape[2]*2:idx*glcm.shape[2]*2+glcm.shape[2]*2] = np.concatenate((a, r))

    else:
        features = np.zeros(glcm.shape[2] * glcm.shape[3] * len(props))

        for idx, prop in enumerate(props):
            f = graycoprops(glcm, prop)
            # f展平为[d1θ1, d2θ1, ..., d1θ2, d2θ2, ...]
            features[idx * glcm.shape[2] * glcm.shape[3]:
                     idx * glcm.shape[2] * glcm.shape[3] + glcm.shape[2] * glcm.shape[3]] = f.flatten('F')

    return features[None, ...]


def get_glcm_feature_names(distances, angles, props):
    glcm_feature_names = []
    for prop in props:
        for angle in angles:
            for distance in distances:
                glcm_feature_names.append("glcm_{:03d}px_{:03d}deg_{}".format(distance, int(angle * 180 / math.pi), prop))
    return glcm_feature_names



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

            valid = bin_image[bin_image < bins]
            h = np.bincount(
                valid.astype(np.int32),
                minlength=bins
            ).astype(np.float32)
            h /= h.sum()

            # values, counts = np.unique(bin_image, return_counts=True)
            #
            # h = np.zeros((1, bins))
            # for k in range(len(values)):
            #     # 过滤掉0像素值，为非法值bins，正常为[0,bins-1]
            #     if not values[k] == bins:
            #         h[0, int(values[k])] = counts[k]
            # h = h/np.sum(h)
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

def preprocess_texture_image(image, standardize_lbp_image,standardize_glcm_image):
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    mask = image == 0
    low, high = None, None
    standardize=standardize_lbp_image and standardize_glcm_image
    if standardize:
        foreground = image[~mask]

        texture_mean = foreground.mean()
        texture_std = foreground.std()

        low = round(texture_mean - 3.1 * texture_std)
        high = round(texture_mean + 3.1 * texture_std)

        image = np.clip(image, low, high)

    return image, mask,low,high