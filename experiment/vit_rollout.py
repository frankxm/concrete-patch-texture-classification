import torch
import numpy as np
from models.efficientformer import Attention

def rollout(attentions, discard_ratio, head_fusion,use_clstoken):
    # result 是单位矩阵，表示“初始总 attention = identity” [49,49]
    result = torch.eye(attentions[0].size(-1),device=attentions[0].device)
    masks=[]
    with torch.no_grad():
        for attention in attentions:
            attention=attention.detach()
            # 平均容易造成注意力稀释
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            # 注意力更加聚焦，但容易放大噪声
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            #     突出所有头都关注的地方
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # Drop the lowest attentions, but
            # don't drop the class token
            if use_clstoken:
                indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1),device=attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            # a 是当前层处理后的注意力矩阵（[N, N]）。result 保存 之前所有层累积的 attention。 当前层乘以前层累积
            result = torch.matmul(a, result)

            if use_clstoken:
                # Look at the total attention between the class token,
                # and the image patches
                # 这里默认是vit，拥有cls_token，意思是取第 0 个 batch，第 0 个 token（CLS token）对所有 patch（不包括cls_token) 的 attention  [1,49,49]
                mask = result[0, 0, 1:]
            else:
                # 这里因为没有cls_token，所以固定每个列j，表示对于每个token，获取其他所有token对他的总的注意力
                mask = result[0].sum(dim=0)  # shape: [N] = [49]

            # In case of 224x224 image, this brings us from 196 to 14
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(width, width).cpu().numpy()
            mask = mask / np.max(mask+ 1e-8)



            masks.append(mask)
    return masks

#
# def grad_rollout(attentions, gradients, discard_ratio, head_fusion,use_cls_token):
#
#
#     result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
#     masks = []
#
#     with torch.no_grad():
#         for attention, grad in zip(attentions, gradients):
#
#             # 平均容易造成注意力稀释
#             if head_fusion == "mean":
#                 attention_heads_fused = (attention * grad).mean(axis=1)
#                 attention_heads_fused = attention_heads_fused - attention_heads_fused.min(dim=-1, keepdim=True)[0]
#                 attention_heads_fused = attention_heads_fused / (
#                             attention_heads_fused.max(dim=-1, keepdim=True)[0] + 1e-8)
#             # 注意力更加聚焦，但容易放大噪声
#             elif head_fusion == "max":
#                 attention_heads_fused = (attention * grad).max(axis=1)[0]
#             #     突出所有头都关注的地方
#             elif head_fusion == "min":
#                 attention_heads_fused = (attention * grad).min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"
#
#             attention_heads_fused[attention_heads_fused < 0] = 0
#
#
#
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
#             # Drop the lowest attentions, but
#             # don't drop the class token
#             if use_cls_token:
#                 indices = indices[indices != 0]
#             flat[0, indices] = 0
#
#             # 单位矩阵 + 归一化
#             I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
#             a = (attention_heads_fused + 1.0 * I) / 2
#             a = a / a.sum(dim=-1)
#             result = torch.matmul(a, result)
#
#             # mask
#             if use_cls_token:
#                 mask = result[0, 0, 1:]
#             else:
#                 mask = result[0].sum(dim=0)
#
#             width = int(mask.size(-1) ** 0.5)
#             mask = mask.reshape(width, width).cpu().numpy()
#             mask = mask / (mask.max() + 1e-8)
#             masks.append(mask)
#
#     return masks

class VITAttentionRollout:
    def __init__(self, model, head_fusion="mean",discard_ratio=0.9,use_grad=False,target_category=0,use_clstoken=False,device='cpu'):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.use_grad = use_grad
        self.target_category = target_category
        self.use_clstoken = use_clstoken
        self.device =device
        self.model.to(self.device)


        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
            # if attention_layer_name in name:
            # hook 的触发顺序 = 前向传播中模块实际执行的顺序
                module.register_forward_hook(self.get_attention)
                # if self.use_grad:
                #     module.register_backward_hook(self.get_attention_gradient)


        self.attentions = []
        self.attention_gradients = []

    # def get_attention(self, module, input, output):
    #     self.attentions.append(output.cpu())
    # pytorch hook固定参数为三，模块，输入，输出
    def get_attention(self, module,input, output):
        if hasattr(module, "attn_visual"):
            self.attentions.append(module.attn_visual)

    #
    # def get_attention_gradient(self, module, grad_input, grad_output):
    #     if hasattr(module, "attn_visual"):
    #         self.attention_gradients.append(grad_output[0])

    def __call__(self, input_tensor):
        if not self.use_grad:
            self.attentions = []
            with torch.no_grad():
                # 模型前向传播结束后，self.attentions 会按顺序保存所有触发 hook 的 attention。
                output = self.model(input_tensor)

            return rollout(self.attentions, self.discard_ratio, self.head_fusion,self.use_clstoken)
        # else:
        #     self.attentions = []
        #     self.attention_gradients = []
        #     self.model.zero_grad()
        #     output = self.model(input_tensor, 'train')
        #     category_mask = torch.zeros_like(output, device=self.device)
        #     category_mask[:, self.target_category] = 1
        #     loss = (output * category_mask).sum()
        #     loss.backward(retain_graph=True)
        #
        #     # 获取每层 attn_visual 对 loss 的梯度
        #     for attn in self.attentions:
        #         grad = torch.autograd.grad(loss, attn, retain_graph=True)[0]
        #         self.attention_gradients.append(grad)
        #
        #     return grad_rollout(self.attentions, self.attention_gradients,
        #                         self.discard_ratio,self.head_fusion, self.use_clstoken)
