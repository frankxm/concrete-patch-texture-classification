"""
EfficientFormer
"""
import os
import copy
import torch
import torch.nn as nn

from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# timm.layers.helpers
from timm.models.layers.helpers import to_2tuple
from torch.cuda.amp import autocast

EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}

EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.N = resolution ** 2
        self.N2 = self.N
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)


        return x


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Flat(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)

        x = self.norm1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.norm2(x)

        x = self.drop(x)
        return x


class Meta3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:

            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(Meta3D(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
        else:
            blocks.append(Meta4D(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())

    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.3, drop_path_rate=0.3,
                 use_layer_scale=True, layer_scale_init_value=1e-6,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,use_amp=False,
                 **kwargs):
        super().__init__()
        self.embed_dims=embed_dims
        self.amp=use_amp
        self.softmax = torch.nn.Softmax(dim=1)

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()



    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    # def init_weights(self, pretrained=None):
    #     # logger = get_root_logger()
    #     if self.init_cfg is None and pretrained is None:
    #         logger.warn(f'No pre-trained weights for '
    #                     f'{self.__class__.__name__}, '
    #                     f'training start from scratch')
    #         pass
    #     else:
    #         assert 'checkpoint' in self.init_cfg, f'Only support ' \
    #                                               f'specify `Pretrained` in ' \
    #                                               f'`init_cfg` in ' \
    #                                               f'{self.__class__.__name__} '
    #         if self.init_cfg is not None:
    #             ckpt_path = self.init_cfg['checkpoint']
    #         elif pretrained is not None:
    #             ckpt_path = pretrained
    #
    #         ckpt = _load_checkpoint(
    #             ckpt_path, logger=logger, map_location='cpu')
    #         if 'state_dict' in ckpt:
    #             _state_dict = ckpt['state_dict']
    #         elif 'model' in ckpt:
    #             _state_dict = ckpt['model']
    #         else:
    #             _state_dict = ckpt
    #
    #         state_dict = _state_dict
    #         missing_keys, unexpected_keys = \
    #             self.load_state_dict(state_dict, False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x,step='train',return_feature=False):
        with autocast(enabled=self.amp):
            x = self.patch_embed(x)
            x = self.forward_tokens(x)
            if self.fork_feat:
                # otuput features of four stages for dense prediction
                return x
            x = self.norm(x) #[b,49,512]
            pooled=x.mean(-2)# [B, embed_dim] [b,512]

            if return_feature:
                # 返回特征向量，不经过 head
                return pooled

            if self.dist:
                cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
                if not self.training:
                    cls_out = (cls_out[0] + cls_out[1]) / 2
            else:
                cls_out = self.head(x.mean(-2))

            if step == 'train':
                return cls_out
            elif step=='prediction':
                return self.softmax(cls_out)



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def efficientformer_l1(num_classes, use_amp, **kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,num_classes=num_classes,use_amp=use_amp,distillation=False,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformer_l3(num_classes, use_amp, **kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer_width['l3'],
        downsamples=[True, True, True, True],
        vit_num=4,num_classes=num_classes,use_amp=use_amp,distillation=False,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformer_l7(pretrained=False, **kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer_width['l7'],
        downsamples=[True, True, True, True],
        vit_num=8,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model
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

def measure_latency_cpu(model, batch_size=1):
    device = torch.device("cpu")
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)
    model.to(device)

    repetitions = 3000
    timings = np.zeros((repetitions, 1))

    # CPU WARM-UP
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()

            curr_time = (end_time - start_time) * 1000  # 转换成 ms
            timings[rep] = curr_time

        mean_time = np.mean(timings)
        std_time = np.std(timings)

        print("Mean Inference Time on CPU: {:.4f} ms".format(mean_time))
        print("Standard Deviation: {:.4f} ms".format(std_time))

def measure_latency(model,batch_size=1):
    device = torch.device('cuda:0')
    # 创建随机输入张量并移动到 GPU

    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)


    model.to(device)  # 指定使用 CUDA GPU 设备

    # 创建 CUDA 事件对象
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # 设置重复次数
    repetitions = 3000
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP：预热 GPU
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # 等待 GPU 同步
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        # 计算平均推理时间和标准差
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        # 打印输出结果
        print("Mean Inference Time: {:.4f} ms".format(mean_time))
        print("Standard Deviation: {:.4f} ms".format(std_time))
def find_max_batchsize(model, input_size, device='cuda', dtype=torch.float, max_trial=16384):

    model.to(device)
    model.eval()

    low = 1
    high = max_trial
    max_batch = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            dummy_input = torch.randn((mid, *input_size), device=device, dtype=dtype)
            with torch.no_grad():
                _ = model(dummy_input)
            # 如果成功，不断尝试更大的 batch size
            max_batch = mid
            low = mid + 1
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # OOM 出现，减小 batch size
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e
    return max_batch
if __name__ == '__main__':
    from thop import profile
    import numpy as np
    import time
    model = efficientformer_l3(num_classes=4,use_amp=False)
    # print(model)
    # 30.38M
    print(f"Total number of parameters: {count_parameters_in_proper_unit(model)}")


    input = torch.randn(1, 3, 224, 224)

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    # input_size = (3, 224, 224)
    # max_batch = find_max_batchsize(model, input_size)
    # print("Maximum batch size:", max_batch)
    measure_latency(model)
    # measure_latency(model, max_batch)

#
# def measure_latency(model, device, batch_size=1, repetitions=1000):
#     """
#     测量单个设备上模型的 latency / FPS
#     """
#     # 创建随机输入
#     dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
#     model.to(device)
#     model.eval()
#
#     timings = []
#
#     # CPU 或 GPU 预热
#     with torch.no_grad():
#         for _ in range(50):
#             _ = model(dummy_input)
#
#     # 测量 latency
#     with torch.no_grad():
#         for _ in range(repetitions):
#             start_time = time.perf_counter()
#             _ = model(dummy_input)
#             end_time = time.perf_counter()
#             timings.append((end_time - start_time) * 1000)  # ms
#
#     timings = np.array(timings)
#     mean_time = np.mean(timings)
#     std_time = np.std(timings)
#     fps = 1000 / mean_time  # batch=1
#     return mean_time, std_time, fps, timings
#
#
# # -----------------------------
# # 配置不同实验条件
# # -----------------------------
# models = {"EfficientFormer": efficientformer_model,
#           "VGG": vgg_model}
#
# devices = ["cpu", "cuda"]  # GPU/CPU
# cpu_cores_list = [1, 4, 8, 16]  # 只对 CPU 有效
# batch_sizes = [1, 2, 4]  # 可扩展
#
# results = []
#
# for model_name, model in models.items():
#     for device_name in devices:
#         if device_name == "cpu":
#             for num_cores in cpu_cores_list:
#                 torch.set_num_threads(num_cores)
#                 for batch_size in batch_sizes:
#                     mean_time, std_time, fps, timings = measure_latency(
#                         model, torch.device("cpu"), batch_size=batch_size
#                     )
#                     results.append({
#                         "Model": model_name,
#                         "Device": "CPU",
#                         "CPU_Cores": num_cores,
#                         "Batch_Size": batch_size,
#                         "Mean_Latency_ms": mean_time,
#                         "Std_Latency_ms": std_time,
#                         "FPS": fps
#                     })
#         else:
#             # GPU 情况不需要设置核心数
#             for batch_size in batch_sizes:
#                 mean_time, std_time, fps, timings = measure_latency(
#                     model, torch.device("cuda"), batch_size=batch_size
#                 )
#                 results.append({
#                     "Model": model_name,
#                     "Device": "GPU",
#                     "CPU_Cores": None,
#                     "Batch_Size": batch_size,
#                     "Mean_Latency_ms": mean_time,
#                     "Std_Latency_ms": std_time,
#                     "FPS": fps
#                 })
#
# df = pd.DataFrame(results)
# df.to_csv("latency_fps_results.csv", index=False)
# print(df)
# output_csv = "latency_fps_results.csv"
# df.to_csv(output_csv, index=False)
# print(f"Latency/FPS results saved to {output_csv}")


