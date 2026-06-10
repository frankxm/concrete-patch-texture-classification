import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from thop import profile
import numpy as np
import time
import os

class ConvClassifier(nn.Module):
# 只使用了 一个卷积层，并且 out_channels=1，没有学习到丰富的通道信息。也没有卷积堆叠来提取更复杂的空间结构。最终输出维度从 [B, 1, 532, 1] → [B, 1, 527, 1]，信息压缩不明显，几乎没做什么 feature extraction。
    def __init__(self, input_size=532, num_classes=4, conv_size=6, dropout=0.3,use_amp=False):
        super(ConvClassifier, self).__init__()
        self.amp=use_amp
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(conv_size, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size - conv_size + 1, num_classes)  # 计算卷积后大小
        self.softmax = nn.Softmax(dim=1)  # 适用于多分类任务

    def forward(self, x,step='train'):
        with autocast(enabled=self.amp):
            x = self.conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.flatten(x)
            logits = self.fc(x)
            if step == 'train':
                return logits
            elif step == 'prediction':
                return self.softmax(logits)
        return x


# 用来验证multimodal 的texture分支
class ConvTextureEncoder_ablation(nn.Module):
    def __init__(self, out_channels=32, dropout=0.3,fused_dim=512, num_classes=4,use_amp=False):
        super(ConvTextureEncoder_ablation, self).__init__()
        self.output_dim = fused_dim
        self.amp=use_amp
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),  # [B, 1, 532] -> [B, 8, 532]
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, padding=2,dilation=2),  # -> [B, 16, 532]
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, out_channels, kernel_size=3, padding=1),  # normal conv # -> [B, 32, 532]
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(8),  # -> [B, 32, 8]
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),  # -> [B, 32,8]->[B,32*8]
            nn.Dropout(dropout),
            nn.Linear(out_channels * 8, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Linear(self.output_dim , num_classes)

    def forward(self, x,step='train'):
        x = x.squeeze(-1)# [B, 1, 532, 1] → [B, 1, 532]
        with autocast(enabled=self.amp):
            x = self.conv(x) # → [B, C, 8]
            logits = self.mlp(x) #→ [B, fused_dim]

            logits_head = self.head(logits)

            if step == 'train':
                return logits_head
            elif step == 'prediction':
                return self.softmax(logits_head)


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


def measure_latency_cpu(model, batch_size=64):
    device = torch.device("cpu")

    dummy_input = torch.randn(batch_size, 1, 532, 1, dtype=torch.float).to(device)

    model.to(device)

    repetitions = 500
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

        # 计算平均推理时间和标准差
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        # 打印输出结果
        print("Mean Inference Time: {:.4f} ms".format(mean_time))
        print("Standard Deviation: {:.4f} ms".format(std_time))
        # 计算吞吐量 image/ms
        print('batchsize:', batch_size)
        print('np.sum(timings):', np.sum(timings))
        Throughput = (repetitions * batch_size) * 1000.0 / np.sum(timings)
        print('Final Throughput:', Throughput)


def measure_latency(model, batch_size=1):
    device = torch.device('cuda:0')
    # 创建随机输入张量并移动到 GPU

    dummy_input = torch.randn(batch_size, 1, 532, 1, dtype=torch.float).to(device)

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
        # 计算吞吐量 image/ms
        print('batchsize:', batch_size)
        print('np.sum(timings):', np.sum(timings))
        Throughput = (repetitions * batch_size) * 1000.0 / np.sum(timings)
        print('Final Throughput:', Throughput)

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x1):
        outs = []

        for m in self.models:
            out = m(x1)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out)

        return torch.stack(outs, dim=0).mean(dim=0)
if __name__ == '__main__':



    # model = ConvClassifier()
    # # print(model)
    # # 976 parameters
    # print(f"Total number of parameters: {count_parameters_in_proper_unit(model)}")
    #
    #
    # input = torch.randn(1, 1, 532, 1)
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops / 1e3:.2f} KFLOPs")
    #
    # measure_latency(model)
    # # measure_latency_cpu(model)

    net1 = ConvClassifier()
    net2 = ConvClassifier()
    net3 = ConvClassifier()
    net4 = ConvClassifier()
    net5 = ConvClassifier()
    net = EnsembleModel([net1, net2, net3, net4, net5])
    measure_latency(net)
    # measure_latency_cpu(model)



