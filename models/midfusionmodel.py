import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from thop import profile
from models.efficientformer import efficientformer_l3
import numpy as np
import time
import os



class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, vis_feat, tex_feat):
        vis_feat = F.layer_norm(vis_feat, vis_feat.shape[1:])
        tex_feat = F.layer_norm(tex_feat, tex_feat.shape[1:])

        gate = self.gate(torch.cat([vis_feat, tex_feat], dim=1))  # [B, D]
        fused = vis_feat + self.scale*gate * tex_feat
        # tex_feat_gate=self.scale*gate * tex_feat
        # fused = torch.cat([vis_feat, tex_feat_gate], dim=1)
        return fused, gate, self.scale


# 法2 修改conv MLP，层数加深
class ConvTextureEncoder(nn.Module):
    def __init__(self, out_channels=32, dropout=0.3,use_amp=False,fused_dim=512,input_dim=532):
        super(ConvTextureEncoder, self).__init__()
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

            nn.AdaptiveAvgPool1d(8),# -> [B, 32, 8]
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),  # -> [B, 32,8]->[B,32*8]
            nn.Dropout(dropout),
            nn.Linear(out_channels * 8, fused_dim),
            # nn.LayerNorm(fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.squeeze(-1)# [B, 1, 532, 1] → [B, 1, 532]
        with autocast(enabled=self.amp):
            x = self.conv(x) # → [B, C, 8]
            logits = self.mlp(x)  # → [B, fused_dim]
            return logits

#

class MidFusionModel(nn.Module):
    def __init__(self, efficientformer: nn.Module, 
                 texture_branch : nn.Module,
                 fused_hidden_size=512, 
                 num_classes=4, dropout=0.5,use_amp=False):
        super().__init__()
        self.efficientformer = efficientformer
        self.texture_branch  = texture_branch
        self.amp=use_amp
        # gateconcat分类头
        # self.head = nn.Sequential(
        #     nn.Linear( efficientformer.embed_dims[-1] + texture_branch.output_dim, fused_hidden_size),
        #     nn.BatchNorm1d(fused_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fused_hidden_size, num_classes)
        # )

        dim = efficientformer.embed_dims[-1]
        self.fusion = GatedFusion(dim)
        # gatedfusion
        self.head = nn.Sequential(
            nn.Linear(dim, fused_hidden_size),
            nn.BatchNorm1d(fused_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_hidden_size, num_classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)


        # self.vis_norm = nn.LayerNorm(efficientformer.embed_dims[-1])
        # self.tex_norm = nn.LayerNorm(texture_branch.output_dim)



    def forward(self, image, texture,step='train'):
        with autocast(enabled=self.amp):
            vis_feat = self.efficientformer(image, return_feature=True)  # [B, 512]
            tex_feat = self.texture_branch (texture)  # [B, D']
            # ##### visual branch主导
            # # # 一阶统计 mean ,二阶std
            # print("vis mean:", vis_feat.mean().item())
            # print("vis std :", vis_feat.std().item())
            # print("tex mean:", tex_feat.mean().item())
            # print("tex std :", tex_feat.std().item())
            # # L2 norm
            # vis_norm = vis_feat.norm(dim=1).mean()
            # tex_norm = tex_feat.norm(dim=1).mean()
            # print(vis_norm, tex_norm)

            ### layernorm
            # vis_feat = self.vis_norm(vis_feat)
            # tex_feat = self.tex_norm(tex_feat)

            # print("vis mean:", vis_feat.mean().item())
            # print("vis std :", vis_feat.std().item())
            # print("tex mean:", tex_feat.mean().item())
            # print("tex std :", tex_feat.std().item())
            # vis_norm = vis_feat.norm(dim=1).mean()
            # tex_norm = tex_feat.norm(dim=1).mean()
            # print(vis_norm, tex_norm)


            # fused = torch.cat([vis_feat, tex_feat], dim=1)
            fused,gate, scale  = self.fusion(vis_feat, tex_feat)  # [B, 512]
            logits = self.head(fused)
            if step == 'train':
                return logits
            elif step=='prediction':
                return self.softmax(logits)

            # # 双头
            # shared_feat = self.shared(fused)
            # logits_cls = self.head_cls(shared_feat)  # [B, 4]
            # logits_ecrase = self.head_ecrase(shared_feat)  # [B, 1]
            #
            # if step == 'train':
            #     return logits_cls,logits_ecrase,gate,scale
            # elif step=='prediction':
            #     prob_cls = self.softmax(logits_cls)
            #     prob_ecrase = torch.sigmoid(logits_ecrase)
            #     return prob_cls, prob_ecrase




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
    dummy_input2 = torch.randn(batch_size, 1, 532, 1, dtype=torch.float).to(device)

    model.to(device)

    repetitions = 500
    timings = np.zeros((repetitions, 1))

    # CPU WARM-UP
    with torch.no_grad():
        model.eval()
        for _ in range(20):
            _ = model(dummy_input, dummy_input2)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        model.eval()
        for rep in range(repetitions):
            start_time = time.perf_counter()
            _ = model(dummy_input, dummy_input2)
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
    dummy_input2 = torch.randn(batch_size, 1, 532, 1, dtype=torch.float).to(device)

    model.to(device)  # 指定使用 CUDA GPU 设备

    # 创建 CUDA 事件对象
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # 设置重复次数
    repetitions = 3000
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP：预热 GPU
    with torch.no_grad():
        model.eval()
        for _ in range(100):
            _ = model(dummy_input, dummy_input2)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        model.eval()
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input, dummy_input2)
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

    def forward(self, x1, x2):
        outs = []

        for m in self.models:
            out = m(x1, x2)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out)

        return torch.stack(outs, dim=0).mean(dim=0)

if __name__ == '__main__':
    # # ###### interop->intraop->1.omp 2.MKL->cpu kernel execution->cudnn benchmark->gpu kernel
    # # CPU层（防止线程爆炸）控制cpu推理时，PyTorch到底允许多少CPU线程参与计算
    # # CPU线程控制 openmp线程数 :PyTorch CPU ops（部分）NumPyOpenCV 一些底层算子（比如 layernorm / reduce
    # # 不是最多只能用8个cpu
    # os.environ["OMP_NUM_THREADS"] = "8"
    # # Intel MKL（矩阵库）线程数：torch.matmul  nn.Linear  NumPy（如果 linked MKL）
    # os.environ["MKL_NUM_THREADS"] = "8"
    # # PyTorch intra-op 线程池，控制一个operation算子内部开多少线程
    # # PyTorch CPU threading was limited to 8 threads for runtime consistency.
    # torch.set_num_threads(8)
    # # 算子之间的并行 控制几个op并行
    # torch.set_num_interop_threads(1)
    #
    # # # GPU层（卷积优化）允许 cuDNN 自由选最快算法 使用场景：inference benchmark 输入尺寸固定  追求速度  单一模型推理
    # torch.backends.cudnn.benchmark = True
    #
    # net1 = efficientformer_l3(num_classes=4, use_amp=False)
    # net2 = ConvTextureEncoder(use_amp=False)
    # net = MidFusionModel(net1, net2, num_classes=4, use_amp=False)
    # # print(net)
    #
    # print(f"Total number of parameters: {count_parameters_in_proper_unit(net)}")
    # #31.03 M
    # input_texture = torch.randn(1, 1, 532, 1)
    # input_image = torch.randn(1, 3, 224, 224)
    # flops, params = profile(net, inputs=(input_image,input_texture))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    # print(f"Params: {params / 1e6:.2f} M")
    #
    # measure_latency(net)
    # # measure_latency_cpu(net)

    # ###### interop->intraop->1.omp 2.MKL->cpu kernel execution->cudnn benchmark->gpu kernel
    # CPU层（防止线程爆炸）控制cpu推理时，PyTorch到底允许多少CPU线程参与计算
    # CPU线程控制 openmp线程数 :PyTorch CPU ops（部分）NumPyOpenCV 一些底层算子（比如 layernorm / reduce
    # 不是最多只能用8个cpu
    os.environ["OMP_NUM_THREADS"] = "8"
    # Intel MKL（矩阵库）线程数：torch.matmul  nn.Linear  NumPy（如果 linked MKL）
    os.environ["MKL_NUM_THREADS"] = "8"
    # PyTorch intra-op 线程池，控制一个operation算子内部开多少线程
    # PyTorch CPU threading was limited to 8 threads for runtime consistency.
    torch.set_num_threads(8)
    # 算子之间的并行 控制几个op并行
    torch.set_num_interop_threads(1)

    # # GPU层（卷积优化）允许 cuDNN 自由选最快算法 使用场景：inference benchmark 输入尺寸固定  追求速度  单一模型推理
    torch.backends.cudnn.benchmark = True

    net_eff = efficientformer_l3(num_classes=4, use_amp=False)
    net_conv = ConvTextureEncoder(use_amp=False)
    net1 = MidFusionModel(net_eff, net_conv, num_classes=4, use_amp=False)
    net2 = MidFusionModel(net_eff, net_conv, num_classes=4, use_amp=False)
    net3 = MidFusionModel(net_eff, net_conv, num_classes=4, use_amp=False)
    net4 = MidFusionModel(net_eff, net_conv, num_classes=4, use_amp=False)
    net5 = MidFusionModel(net_eff, net_conv, num_classes=4, use_amp=False)
    net = EnsembleModel([net1, net2, net3, net4, net5])

    measure_latency(net)
    # measure_latency_cpu(net)