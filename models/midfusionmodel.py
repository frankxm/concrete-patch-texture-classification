import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 法2 修改conv MLP，层数加深
class ConvTextureEncoder(nn.Module):
    def __init__(self, out_channels=32, dropout=0.3,use_amp=False,fused_dim=512):
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

    def forward(self, x):
        x = x.squeeze(-1)# [B, 1, 532, 1] → [B, 1, 532]
        with autocast(enabled=self.amp):
            x = self.conv(x) # → [B, C, 8]
            logits = self.mlp(x)  # → [B, fused_dim]
            return logits

class MidFusionModel(nn.Module):
    def __init__(self, efficientformer: nn.Module, 
                 texture_branch : nn.Module,
                 fused_hidden_size=512, 
                 num_classes=4, dropout=0.5,use_amp=False):
        super().__init__()
        self.efficientformer = efficientformer
        self.texture_branch  = texture_branch
        self.amp=use_amp
        # 分类头
        self.head = nn.Sequential(
            nn.Linear(efficientformer.embed_dims[-1] + texture_branch.output_dim , fused_hidden_size),
            nn.BatchNorm1d(fused_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_hidden_size, num_classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, image, texture,step='train'):
        with autocast(enabled=self.amp):
            vis_feat = self.efficientformer(image, return_feature=True)  # [B, 512]
            tex_feat = self.texture_branch (texture)  # [B, D']

            # 拼接
            fused = torch.cat([vis_feat, tex_feat], dim=1)  # [B, D+D']

            # 分类头
            logits = self.head(fused)

            if step == 'train':
                return logits
            elif step=='prediction':
                return self.softmax(logits)

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
    from thop import profile
    from models.efficientformer import efficientformer_l3

    net1 = efficientformer_l3(num_classes=4, use_amp=False)
    net2 = ConvTextureEncoder(use_amp=False)
    net = MidFusionModel(net1, net2, num_classes=4, use_amp=False)
    print(net)
    #31.03 M
    input_texture = torch.randn(1, 1, 532, 1)
    input_image = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input_image,input_texture))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")