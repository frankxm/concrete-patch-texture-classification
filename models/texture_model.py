import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.nn import Module as NNModule
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





# # 法2 修改conv MLP，层数加深
# class ConvClassifier(nn.Module):
#     def __init__(self, input_size=532, num_classes=4, conv_size=6, dropout=0.3,use_amp=False):
#         super(ConvClassifier, self).__init__()
#         self.amp=use_amp
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 8, kernel_size=5, padding=2),  # [B, 1, 532] -> [B, 8, 532]
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#
#             nn.Conv1d(8, 16, kernel_size=3, padding=1),  # -> [B, 16, 532]
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#
#             nn.AdaptiveAvgPool1d(1),  # -> [B, 16, 1]
#         )
#
#         self.mlp = nn.Sequential(
#             nn.Flatten(),  # -> [B, 16]
#             nn.Dropout(dropout),
#             nn.Linear(16, 64),  # 扩展维度
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x,step='train'):
#         x = x.squeeze(-1)
#         with autocast(enabled=self.amp):
#             x = self.conv(x)  # -> [B, 16, 1]
#             logits = self.mlp(x)  # -> [B, num_classes]
#             if step == 'train':
#                 return logits
#             elif step == 'prediction':
#                 return self.softmax(logits)
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


    model = ConvClassifier()
    print(model)
    # 976 parameters
    print(f"Total number of parameters: {count_parameters_in_proper_unit(model)}")


    input = torch.randn(1, 1, 532, 1)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e3:.2f} KFLOPs")



