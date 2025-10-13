# -*- coding: utf-8 -*-

"""
    The training utils module
    ======================

    Use it to during the training stage.
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import training_pixel_metrics as p_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 4):

        super(CustomCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor, ME:torch.Tensor) -> torch.Tensor:

        contains_value_4 = (target == 4).any().cpu().item()
        print(contains_value_4)

        mask = (target != self.ignore_index)
        # reduction=none保证返回的是个矩阵代表每个像素点的交叉熵值，默认情况是一个标量值
        # cross_entropy 函数不会期望 target 是独热编码的形式。它希望 target 是整数索引，即每个像素点只是该像素所属的类别索引，而不是一个多维的 one-hot 向量
        # pred 是网络的预测输出，通常是 (batch_size, num_classes, height, width)。它表示每个像素点属于各个类别的概率分布（未经过 softmax 处理，因为 cross_entropy 内部会处理 log-softmax）。
        # ignore_index 的作用是告诉 cross_entropy 函数对某个特定类别索引的像素点不进行损失计算
        loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        # Softmax 已经包含在 cross_entropy 中：F.cross_entropy 内部会先做 softmax，然后取对数，再进行负对数似然损失的计算。两次softmax 会导致错误的计算。直接对 softmax 输出再取对数，会因为此时的概率数值很小了，从而导致数值下溢或者溢出
        # 对 softmax 输出的概率分布取对数  nll_loss 期望输入的是 log概率，而不是softmax概率本身 需要极端值 易发生nan
        # log_pred = torch.log(torch.clamp(pred, min=1e-10))
        # # 使用 F.nll_loss 计算损失
        # loss = F.nll_loss(log_pred, target, reduction='none',ignore_index=self.ignore_index)
        loss = loss * mask.float()

        weighted_loss = torch.zeros_like(loss)
        # 转独热编码时需要把4限制为3，否则出错，虽然变成了3但是有masque即可保证该部分不算
        MP=F.one_hot(target.clamp(0, self.num_classes - 1),num_classes=self.num_classes).permute(0,3,1,2).contiguous().float()

        for c in range(self.num_classes):
            MPc = MP[:, c, :, :]
            MEc = (ME == 0).float()
            current_MEc = MEc * MPc
            current_MEc = torch.clamp(current_MEc, min=0, max=1)
            # 计算加权损失，只考虑当前类
            M_autre = torch.clamp(MPc - current_MEc, min=0, max=1)
            weighted_loss += (0.5 * M_autre + 1.0 * current_MEc) * loss
            if torch.isnan(weighted_loss).any():
                print("Error: weighted_loss contains NaN values")

        return weighted_loss.sum() / mask.sum().float()

class Diceloss(nn.Module):
    """
    The Diceloss class is used during training.
    """

    def __init__(self, num_classes: int,weights: torch.Tensor = None):
        """
        Constructor of the Diceloss class.
        :param num_classes: The number of classes involved in the experiment.
        """
        super(Diceloss, self).__init__()
        self.num_classes = num_classes

        # 设置权重，默认情况下，每个类别的权重都是1
        if weights is None:
            self.weights = torch.ones(num_classes)
        else:
            self.weights = weights

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        # pred此时大小为(batchsize,classe,height,width)
        # one_hot函数将每个像素点的类别转换为one-hot编码，返回一个形状为(batchsize, height, width, classe)的张量,正确的类别为1，非正确的0
        # permute维度重新排列
        # 这一步确保张量在内存中是连续存储的。虽然permute改变了张量的视图（view），但它不一定会改变张量在内存中的布局。
        # 使用contiguous可以强制张量在内存中是连续的，这在某些操作（如view、reshape）中是必要的。
        label = (
            nn.functional.one_hot(target, num_classes=self.num_classes)
                .permute(0, 3, 1, 2)
                .contiguous()
        )

        smooth = 1.0
        # # 确保数据是连续存储的（通过 contiguous()）后，再将其展平为一维。view(-1)将预测 pred 和标签 label 展平成一维向量，使得可以在所有像素上计算重叠区域的交集和并集。
        # iflat = pred.contiguous().view(-1)
        # tflat = label.contiguous().view(-1)
        # intersection = (iflat * tflat).sum()
        # A_sum = torch.sum(iflat * iflat)
        # B_sum = torch.sum(tflat * tflat)

        # item() 的作用是将包含单个值的张量转换为一个普通的 Python 数据类型，例如 int、float 或 bool。当你调用 .any() 检查 target 中是否包含 4 时，返回的是一个包含布尔值的张量（例如 tensor(True) 或 tensor(False)），而不是直接的 Python 布尔值。
        # 使用 .item() 将这个张量转换为 Python 原生的布尔值 True 或 False，
        contains_value_4 = (target == 4).any().cpu().item()
        print(contains_value_4)

        valid_mask = target != 4
        # 增加一个新维度，大小为1
        valid_mask = valid_mask.unsqueeze(1)
        # expand_as在classe维度上将其扩充
        pred_masked = pred[valid_mask.expand_as(pred)].contiguous().view(-1)
        label_masked = label[valid_mask.expand_as(label)].contiguous().view(-1)

        intersection = (pred_masked * label_masked).sum()
        A_sum = torch.sum(pred_masked * pred_masked)
        B_sum = torch.sum(label_masked * label_masked)
        return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))

    # def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     # pred此时大小为(batchsize,classe,height,width)
    #     #     # one_hot函数将每个像素点的类别转换为one-hot编码，返回一个形状为(batchsize, height, width, classe)的张量,正确的类别为1，非正确的0
    #     #     # permute维度重新排列
    #     #     # 这一步确保张量在内存中是连续存储的。虽然permute改变了张量的视图（view），但它不一定会改变张量在内存中的布局。
    #     #     # 使用contiguous可以强制张量在内存中是连续的，这在某些操作（如view、reshape）中是必要的。
    #     label = nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
    #
    #     smooth = 1.0
    #     dice_loss = 0.0
    #
    #     # 对每一类分别计算diceloss，用于乘以对应权重
    #     for c in range(self.num_classes):
    #         # 确保数据是连续存储的（通过 contiguous()）后，再将其展平为一维。view(-1)将预测 pred 和标签 label 展平成一维向量，使得可以在所有像素上计算重叠区域的交集和并集。
    #         iflat = pred[:, c, :, :].contiguous().view(-1)
    #         tflat = label[:, c, :, :].contiguous().view(-1)
    #
    #         intersection = (iflat * tflat).sum()
    #         A_sum = torch.sum(iflat * iflat)
    #         B_sum = torch.sum(tflat * tflat)
    #
    #         dice_c = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)
    #         weighted_dice_c = self.weights[c] * (1 - dice_c)
    #
    #         dice_loss += weighted_dice_c
    #
    #     return dice_loss / self.num_classes





def plot_prediction(output: np.ndarray) -> np.ndarray:

    prediction = np.zeros((output.shape[0], 1, output.shape[2], output.shape[3]))
    for pred in range(output.shape[0]):
        current_pred = output[pred, :, :, :]
        new = np.argmax(current_pred, axis=0)
        new = np.expand_dims(new, axis=0)
        prediction[pred, :, :, :] = new
    return prediction


def display_training(
        output: np.ndarray,
        image: np.ndarray,
        label: np.ndarray,
        writer,
        epoch: int,
        norm_params: list,
        logpath
):

    predictions = plot_prediction(output.cpu().detach().numpy())

    fig, axs = plt.subplots(
        predictions.shape[0],
        3,
        figsize=(10, 3 * predictions.shape[0]),
        gridspec_kw={"hspace": 0.2, "wspace": 0.05},
    )

    for pred in range(predictions.shape[0]):
        current_input = image.cpu().detach().numpy()[pred, :, :, :]
        current_input = current_input.transpose((1, 2, 0))
        # 逆归一化，所以最后保存的图片会和原图有些许不同 因为经过逆归一化处理后的图像数据通常是浮点数类型，而在显示或保存图像时，你可能会将这些数据转换为整型（如 uint8），这种转换可能会引入舍入误差，导致颜色不完全匹配。
        for channel in range(current_input.shape[2]):
            current_input[:, :, channel] = (current_input[:, :, channel] * norm_params["std"][channel] ) + norm_params["mean"][channel]

        # 逆归一化后确保数据范围在 [0, 255]，相比于直接用astype，clip优点是不会使数据过多改变，超过最大值的就定为最大值。
        # 而astype会把原数据缩放到0到255，某些超过范围的值变化会很大，造成数据丢失
        current_input = np.clip(current_input, 0, 255)
        # 归一化到 [0, 255]，确保显示时正确
        current_input_uint8 = (current_input).astype(np.uint8)
        if predictions.shape[0] > 1:
            axs[pred, 0].imshow(current_input_uint8)
            axs[pred, 1].imshow(label.cpu().detach().numpy()[pred, :, :], cmap="gray")
            axs[pred, 2].imshow(predictions[pred, 0, :, :], cmap="gray")

        else:
            axs[0].imshow(current_input_uint8)
            axs[1].imshow(label.cpu()[pred, :, :], cmap="gray")
            axs[2].imshow(predictions[pred, 0, :, :], cmap="gray")
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    if not os.path.exists(os.path.join(logpath,'Images_in_valid')):
        os.makedirs(os.path.join(logpath,'Images_in_valid'))
    # plt.show()
    plt.savefig(os.path.join(logpath,f"Images_in_valid/current_epoch{epoch}.png"))
    writer.add_figure("Image_Label_Prediction", fig, global_step=epoch)



def get_epoch_values(metrics: dict, classes: list, batch: int) -> dict:

    values = {}
    total_tp=0
    total_samples=0
    for channel in classes:
        recall, precision, f1 = p_metrics.iou(metrics["matrix"], classes.index(channel))
        values["recall_" + channel] = round(recall, 4)
        values["precision_" + channel] = round(precision, 4)
        values["f1_" + channel] = round(f1, 4)

        # 累计总的TP和所有样本数
        total_tp += metrics["matrix"][classes.index(channel), classes.index(channel)]
        total_samples += np.sum(metrics["matrix"][classes.index(channel), :])

        # 计算所有类的总准确率
    if total_samples > 0:
        overall_acc = total_tp / total_samples
    else:
        overall_acc = 0
    values["loss"] = metrics["loss"] / batch
    values["overall_acc"]=round(overall_acc,4)
    return values
