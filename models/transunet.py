# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage


import ml_collections
import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        # 每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        # 用于将所有注意力头的输出连接（concat）并映射回原始的隐藏空间。 all_head_size=hidden_size
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
    # 目的是将输入的张量 x 从形状 (batch_size, seq_length, all_head_size) 转换成适合 多头注意力 机制的形状 (batch_size, num_attention_heads, seq_length, attention_head_size)。
    def transpose_for_scores(self, x):
        # 去掉张量 x 的最后一个维度，通常这个维度是 all_head_size，即所有头的特征维度。x.size() 返回的是一个包含所有维度大小的元组，[:-1] 表示去掉最后一个元素。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # new_x_shape 是一个新的形状，它的维度是 (batch_size, seq_length, num_attention_heads, attention_head_size)。通过 view 操作，x 将被重塑为这个形状
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # 改变形状，适应多头注意力
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 计算查询与键的点积，得到注意力得分。查询和键的乘积表明了输入的不同位置之间的相似性。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        # 用注意力概率加权值向量，计算每个位置的加权和，即上下文向量。
        # attention_probs 的形状是 (batch_size, num_attention_heads, seq_length, seq_length)。
        # value_layer 的形状是 (batch_size, num_attention_heads, seq_length, attention_head_size)。
        # context_layer 的形状： (batch_size, num_attention_heads, seq_length, attention_head_size)，它表示了每个位置加权后的值向量。
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, seq_length, num_attention_heads, attention_head_size) .contiguous()：调用 contiguous() 是为了确保输出的张量在内存中是连续的，尤其是在进行重塑（view()）时，它要求输入张量是连续的。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 合并后两个维度为all_head_size = num_attention_heads * attention_head_size。
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer 的形状从 (batch_size, seq_length, all_head_size) 被映射回 (batch_size, seq_length, hidden_size)。
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        # 确保图像大小是 (H, W) 的元组，比如 (224, 224)。
        img_size = _pair(img_size)
        # 在混合模式下，首先是用 ResNet 等 CNN 进行下采样，将输入图像降至较低分辨率（例如 224x224 → 14x14 特征图），然后在此特征图上进行 patch 划分。所得到的 patch 数量和大小依赖于两个因素：
        if config.patches.get("grid") is not None:   # ResNet
            #先 // 16：图像会被 ResNet 下采样 16 倍 ,在这个特征图上再进行划分指定数量的grid
            grid_size = config.patches["grid"]
            # patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            # # 计算在原图上patch实际大小
            # patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            # n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

            patch_size = (
                math.ceil(img_size[0] / 16 / grid_size[0]),
                math.ceil(img_size[1] / 16 / grid_size[1])
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (
                    math.ceil(img_size[0] / patch_size_real[0]) *
                    math.ceil(img_size[1] / patch_size_real[1])
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        #     使用 Conv2d(stride=kernel_size) 相当于把图像分块 patch 并投影到 hidden_size 维度。
        #  H-P/P+1=H/P 卷积操作相当于把特征图分块了，同时把每块的局部信息投影到hidden_size
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 形状为 (1, N_patches, D)，学习的位置编码。
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            # CNN 会输出一个中间特征图 x，用于后续卷积 patch embedding。
            # features 是一个中间 skip-connection 特征（给 decoder 用）。
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #  (B, hidden. n_patches)
        x = x.flatten(2)
        #  (B,  n_patches,hidden)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        # embedding 是模型学习的基础，如果在这一部分就进行正则化，模型更可能从这些基础信息中学习到更加稳健、不会过拟合的特征。
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        # 将 LayerNorm 放在每个子层之前，有助于让每个子层的输入特征更加稳定，避免了可能的梯度消失或爆炸问题。在深层模型中，这一点尤其重要，因为它有助于保留原始信号的尺度，避免了多个层的累积影响。
        # 通过先进行 LayerNorm，模型的每一层输入会更加规范，这有助于加速收敛并提高训练的稳定性。尤其是在大型模型和长序列的任务中，预归一化可能会显著提高性能。
        # Post-LayerNorm（在注意力机制之后进行归一化）会在每个子层的输出进行标准化，而 Pre-LayerNorm 则是对每个子层的输入进行标准化。这种改变使得每个子层接收到更加均衡和稳定的输入，从而有助于训练
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():


            # query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            #
            # query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            # key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            # value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            # out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            # 确保路径分隔符正确
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel").replace("\\", "/")]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel").replace("\\", "/")]).view(self.hidden_size,
                                                                                                    self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel").replace("\\", "/")]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel").replace("\\", "/")]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias").replace("\\", "/")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias").replace("\\", "/")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias").replace("\\", "/")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias").replace("\\", "/")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            # mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            # mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            # mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            # 确保路径分隔符正确
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel").replace("\\", "/")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel").replace("\\", "/")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias").replace("\\", "/")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias").replace("\\", "/")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            # self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            # self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            # self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            # self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale").replace("\\", "/")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias").replace("\\", "/")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale").replace("\\", "/")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias").replace("\\", "/")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        # 是一个 ModuleList，用来存储多个 Block。Block 是 Transformer 中的基本构建单元，每个 Block 包含了自注意力和前馈网络的结构。num_layers 决定了 Encoder 包含的 Block 的数量。每个 Block 都是独立的，深度学习模型通过堆叠（串联）这些 Block 来增强特征学习能力。
        self.layer = nn.ModuleList()
        # 这是对 Encoder 输出进行归一化的层。它通常应用于 Encoder 的输出，以稳定训练过程。LayerNorm 是一种常见的归一化方法，帮助提升模型的训练稳定性，特别是在深层网络中。
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        # layer_block(hidden_states): 对于每个 Block，输入 hidden_states 会经过该 Block 进行处理，并得到新的 hidden_states 和注意力权重（weights）。hidden_states 是经过多个 Transformer 层后得到的中间状态（通常是最后的输出）。
        # self.encoder_norm(hidden_states): 在所有的 Block 都执行完之后，对 hidden_states 进行一次归一化，以获得最终的输出 encoded。
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # embedding_output 是 Patch embedding + position embedding 的结果。
        # features：如果启用 hybrid 模式（使用 ResNet 提取特征），这是 CNN 的中间多尺度特征，用于 skip connection。
        # encoded：ViT encoder 的最终输出。
        # attn_weights：可视化用的 attention 权重（可选）。
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,use_amp=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        # self.decoder = DecoderCup(config)
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config


        self.softmax = torch.nn.Softmax(dim=1)
        self.amp = use_amp

        self.classifier_head = nn.Sequential(
            nn.Linear(self.config['hidden_size'], 1024),  # 对应 Dense(1024)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),  # 对应 Dense(4)
        )

    def forward(self, x,step='train'):
        with autocast(enabled=self.amp):
        # 把灰度图（1个通道）转换成 RGB 图像（3个通道），因为 Transformer 模型通常是为处理 RGB 图像（3通道）设计的。
        # (1, 3, 1, 1) 的意思是：在第 0 维（batch）复制 1 次 → 不变  在第 1 维（channel）复制 3 次 → 从 1 个通道变成 3 个通道   在第 2 和第 3 维（H, W）复制 1 次 → 不变
            if x.size()[1] == 1:
                x = x.repeat(1,3,1,1)
            x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
            # 对 patch 聚合，用于分类头聚合整图信息
            cls_token = x.mean(dim=1)
            logits = self.classifier_head(cls_token)
            # x = self.decoder(x, features)
            # logits = self.segmentation_head(x)
            if step == 'train':
                return logits
            elif step == 'prediction':
                return self.softmax(logits)


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # 加载 patch embedding 的卷积权重
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            # 加载 Transformer 编码器的 layer norm 参数
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            # 加载位置编码
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            # 如果大小一致，直接拷贝
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            # 如果差 1（原来vit结构有 class token，现在去掉了），剪掉再拷贝
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            # 如果尺寸完全不匹配，插值 resize 成新形状再拷贝
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            print(weights.keys())
            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            #  加载 Transformer 编码器所有 block 如果使用了 hybrid 模型（如 ResNet），加载 CNN backbone 的参数
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)




def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.3
    config.transformer.dropout_rate = 0.3

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path ='./experiment/pretrained/R50+ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config
def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path ='./experiment/pretrained/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config



def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        # conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        # conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        # conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)
        #
        # gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        # gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])
        #
        # gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        # gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])
        #
        # gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        # gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel").replace("\\", "/")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel").replace("\\", "/")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel").replace("\\", "/")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale").replace("\\", "/")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias").replace("\\", "/")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale").replace("\\", "/")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias").replace("\\", "/")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale").replace("\\", "/")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias").replace("\\", "/")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            # proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            # proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            # proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel").replace("\\", "/")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale").replace("\\", "/")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias").replace("\\", "/")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


def count_parameters_in_proper_unit(model):
    total_params = sum(p.numel() for p in model.parameters())

    # 根据参数量的大小选择合适的单位
    if total_params >= 1_000_000_000:
        total_params_in_billions = total_params / 1_000_000_000  # 十亿 (B)
        return f"{total_params_in_billions:.2f}B"  # 以十亿为单位
    elif total_params >= 1_000_000:
        total_params_in_million = total_params / 1_000_000  # 百万 (M)
        return f"{total_params_in_million:.2f}M"  # 以百万为单位
    elif total_params >= 1_000:
        total_params_in_thousands = total_params / 1_000  # 千 (K)
        return f"{total_params_in_thousands:.2f}K"  # 以千为单位
    else:
        return f"{total_params} parameters"  # 小于千的直接显示

if __name__ == '__main__':

    config_vit = get_r50_b16_config()
    config_vit.n_classes = 4
    config_vit.patches.grid = (int(224 / 16), int(224 / 16))
    net = VisionTransformer(config_vit, img_size=224, num_classes=4).cuda()
    print(net)
    #  98.68M
    print(f"Total number of parameters: {count_parameters_in_proper_unit(net)}")