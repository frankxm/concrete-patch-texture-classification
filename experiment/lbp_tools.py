import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage.feature.texture import local_binary_pattern


def region_lbp(image, radii, ps, levels=256, standardize=False):

    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    mask = np.where(image == 0)
    if standardize:
        true_pixels_mask = np.where(image > 0)
        texture_mean = np.mean(image[true_pixels_mask])
        texture_std = np.std(image[true_pixels_mask])

        standardized_image = np.copy(image)
        limits = [round(texture_mean - 3.1 * texture_std), round(texture_mean + 3.1 * texture_std)]
        standardized_image[np.where(standardized_image < limits[0])] = limits[0]
        standardized_image[np.where(standardized_image > limits[1])] = limits[1]

        bin_image = np.floor(
            levels * (standardized_image.astype(float) - limits[0]) / (limits[1] - limits[0] + 1)).astype(np.uint16)
    else:
        bin_image = np.floor(levels * image.astype(float) / 256.0).astype(np.uint16)

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(bin_image, cmap='gray')
    plt.title(f'Binned Image(bins={levels})')
    plt.axis('off')
    plt.show()
    processed_image = np.copy(bin_image)
    bin_image[mask] = levels



    lbps = np.zeros((image.shape[0], image.shape[1], len(radii), len(ps)))
    for i, radius in enumerate(radii):
        for j, p in enumerate(ps):
            lbp = local_binary_pattern(bin_image, p, radius, method='default')
            lbp[mask] = 2**p
            lbps[..., i, j] = lbp

    return lbps


def get_lbp_histograms(lbp, bins):

    hs = []
    for i in range(lbp.shape[3]):
        for j in range(lbp.shape[2]):
            current_image = lbp[..., j, i]
            mask_value = np.max(current_image)
            bin_image = np.floor(bins * current_image / mask_value)

            values, counts = np.unique(bin_image, return_counts=True)

            h = np.zeros((1, bins))
            for k in range(len(values)):
                if not values[k] == bins:
                    h[0, int(values[k])] = counts[k]
            h = h/np.sum(h)
            hs.append(h)

            # plt.figure()
            # plt.bar(range(bins), h.flatten(), color='skyblue')
            # plt.xlabel('Bin')
            # plt.ylabel('Normalized Count')
            # plt.title(f'LBP Histogram: radius_index={j}, points_index={i}')
            # plt.show()

    features = np.ravel(np.concatenate(hs))[None, ...]
    return features


def get_lbp_feature_names(radii, ps, bins):
    lbp_feature_names = []
    for p in ps:
        for radius in radii:
            for bin in range(bins):
                lbp_feature_names.append("lbp_radius{:02d}_p{:02d}_bin{:03d}".format(radius, p, bin))
    return lbp_feature_names