import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
from thop import profile
import numpy as np
import time
import os


class CustomVGG19(nn.Module):
    def __init__(self, num_classes: int = 4, use_amp=False):
        super().__init__()
        self.amp=use_amp
        # 加载VGG模型, 不包含分类头
        # vgg19 = models.vgg19(pretrained=False)  # 加载VGG19预训练模型 imagenet-1k 1000类
        vgg19 = models.vgg19_bn(pretrained=False)
        # 特征提取部分（卷积层和池化层）
        self.features = vgg19.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GlobalAveragePooling2D 对应

        # self.classifier = nn.Sequential(
        #
        #     nn.Flatten(),
        #     nn.Linear(512, 1024),  # 对应 Dense(1024)
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, num_classes),  # 对应 Dense(4)
        # )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )


        self.softmax = nn.Softmax(dim=1)
    #     nn.Softmax(dim=1) 通常是用于分类问题中的最后一层，它将输出转换为概率分布。然而，CrossEntropyLoss 组合了 log_softmax 和 nll_loss，因此不需要在最后一层使用 Softmax。如果你使用了 Softmax 和 CrossEntropyLoss，这会导致数值不稳定，因为你会对概率进行重复的变换。


    def forward(self, x,step='train'):
        with autocast(enabled=self.amp):
            x = self.features(x)
            x = self.avgpool(x)
            logits = self.classifier(x)
            if step == 'train':
                return logits
            elif step == 'prediction':
                return self.softmax(logits)
        return x
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
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)
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
if __name__ == "__main__":
    # # ###### interop->intraop->1.omp 2.MKL->cpu kernel execution->cudnn benchmark->gpu kernel
    # # CPU层（防止线程爆炸）
    # # CPU线程控制 openmp线程数 :PyTorch CPU ops（部分）NumPyOpenCV 一些底层算子（比如 layernorm / reduce
    # os.environ["OMP_NUM_THREADS"] = "8"
    # # Intel MKL（矩阵库）线程数：torch.matmul  nn.Linear  NumPy（如果 linked MKL）
    # os.environ["MKL_NUM_THREADS"] = "8"
    # # PyTorch intra-op 线程池，控制一个op多少线程
    # torch.set_num_threads(8)
    # # 算子之间的并行 控制几个op并行
    # torch.set_num_interop_threads(1)
    #
    # # # GPU层（卷积优化）
    # torch.backends.cudnn.benchmark = True
    #
    #
    # # 输入图像大小224
    # model = CustomVGG19()
    # # print(model)
    # # 打印模型的参数总数，以适当单位表示 20.55M
    # print(f"Total number of parameters: {count_parameters_in_proper_unit(model)}")
    #
    # input = torch.randn(1, 3, 224, 224)
    #
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    # print(f"Params: {params / 1e6:.2f} M")
    #
    # measure_latency(model)
    # # measure_latency_cpu(model)

    # ###### interop->intraop->1.omp 2.MKL->cpu kernel execution->cudnn benchmark->gpu kernel
    # CPU层（防止线程爆炸）
    # CPU线程控制 openmp线程数 :PyTorch CPU ops（部分）NumPyOpenCV 一些底层算子（比如 layernorm / reduce
    os.environ["OMP_NUM_THREADS"] = "12"
    # Intel MKL（矩阵库）线程数：torch.matmul  nn.Linear  NumPy（如果 linked MKL）
    os.environ["MKL_NUM_THREADS"] = "12"
    # PyTorch intra-op 线程池，控制一个op多少线程
    torch.set_num_threads(12)
    # 算子之间的并行 控制几个op并行
    torch.set_num_interop_threads(1)

    # # GPU层（卷积优化）
    torch.backends.cudnn.benchmark = True

    # 输入图像大小224
    net1 = CustomVGG19()
    net2 = CustomVGG19()
    net3 = CustomVGG19()
    net4 = CustomVGG19()
    net5 = CustomVGG19()
    net = EnsembleModel([net1, net2, net3, net4, net5])

    measure_latency(net)
    # measure_latency_cpu(model)

