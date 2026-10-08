import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        # 用于把attention 层的shape从 [B, N, C] 转成 [B, C, H, W] 便于 GradCAM 计算
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            # forward hook → 当数据通过这个 layer 前向传播时，把 输出特征图 activation 保存下来
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                # backward hook → 当 loss 反向传播到这个 layer 时，把 梯度 grad 保存下来
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
    # 记录反向传播的梯度，靠近输出层的梯度在前
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        #     detach 只是为了把 tensor 从计算图中分离，方便后续 numpy 计算
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """
    # 对每个 channel 的梯度在空间维度求平均 → 得到权重 越能影响目标类别的 channel 权重大
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss
    # get_cam_image 对每个 channel 做加权求和得到 CAM mask ，即热力图原始mask
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height
    # compute_cam_per_layer 会从 ActivationsAndGradients 获取 特征图 activations 和 梯度 grads
    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            # ReLU 保证 mask >= 0  scale_cam_image 归一化到 [0,1] 并 resize 到原图大小
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result
    # 用输出对某一层特征图的梯度来加权该层特征图。
    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播  得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        # 根据目标类别计算loss
        loss = self.get_loss(output, target_category)


        # loss回传梯度 反向传播
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


# def show_cam_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
#     """ This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.
#
#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :returns: The default image with the cam overlay.
#     """
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255
#
#     if np.max(img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")
#
#     cam = heatmap + img
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET):
    """
    Display GradCAM: outputs original image, heatmap, and overlay image.

    :param img: Original image, HxWx3, float32, range [0,1]
    :param mask: GradCAM mask, HxW, float32, range [0,1]
    :param use_rgb: If True, convert heatmap to RGB
    :param colormap: OpenCV colormap for heatmap
    :return: tuple of (original image, heatmap, overlay)
             all in HxWx3, uint8
    """

    # 原图
    original = np.clip(img, 0, 1)

    # CAM heatmap 将 mask 变成彩色热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255  # 归一化到 [0,1]

    # 扩展 heatmap 通道与原图一致
    if heatmap.shape[:2] != original.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    # 叠加 和原图叠加，生成最终可视化
    overlay = heatmap + original
    overlay = overlay / np.max(overlay)

    # 转 uint8
    original_uint8 = np.uint8(255 * original)
    heatmap_uint8 = np.uint8(255 * heatmap)
    overlay_uint8 = np.uint8(255 * overlay)

    return original_uint8, heatmap_uint8, overlay_uint8

def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img