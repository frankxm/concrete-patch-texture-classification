# -*- coding: utf-8 -*-

import copy
import logging
import os
import sys
import time
import numpy as np
import torch

# 获取当前脚本所在目录（experiment 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from models.inceptionresnetv2 import InceptionResNetV2
from models.efficientformer import efficientformer_l3
from models.custom_vgg19 import CustomVGG19
from models.texture_model import ConvClassifier
from models.transunet import get_r50_b16_config,VisionTransformer
from models.midfusionmodel import MidFusionModel,ConvTextureEncoder
logger = logging.getLogger(__name__)



def weights_init(model):

# 考虑了偏置项和各种层

    if isinstance(model, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            # 如果存在偏置项（bias），则将其初始化为零（nn.init.constant_）
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, torch.nn.BatchNorm2d):
        # 初始化批归一化层（BatchNorm2d）：将权重初始化为1 将偏置初始化为零
        torch.nn.init.constant_(model.weight.data, 1)
        torch.nn.init.constant_(model.bias.data, 0)

def get_models(model_name,no_of_classes,use_amp):
    if model_name=='customvgg19':
        net=CustomVGG19(num_classes=no_of_classes, use_amp=use_amp)

    elif model_name=='efficientformerl3':
        net=efficientformer_l3(num_classes=no_of_classes, use_amp=use_amp)
    elif model_name=='inceptionresnetv2':
        net=InceptionResNetV2(num_classes=no_of_classes, use_amp=use_amp)
    elif model_name=='texture_model':
        net=ConvClassifier(num_classes=no_of_classes, use_amp=use_amp)
    elif model_name=='customtransunet':
        config_vit =get_r50_b16_config()
        config_vit.n_classes = no_of_classes
        net = VisionTransformer(config_vit, num_classes=no_of_classes, use_amp=use_amp)
    elif model_name=='midfusionmodel':
        net1 = efficientformer_l3(num_classes=no_of_classes, use_amp=use_amp)
        net2 = ConvTextureEncoder(use_amp=use_amp)
        net=MidFusionModel(net1,net2,num_classes=no_of_classes, use_amp=use_amp)

    logger.info(f"Creating model: {model_name}")
    return net

def load_network(no_of_classes, use_amp,use_gpu,model_name):

    net=get_models(model_name,no_of_classes,use_amp)


    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logger.info("Running on %s", device)
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs", torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
    return net.to(device)


def restore_model(
    net, optimizer, scaler, model_path,model_name,keep_last: bool = True
):

    starting_time = time.time()
    if not os.path.isfile(model_path):
        logger.error("No model found at %s",  model_path)
        sys.exit()
    else:
        # pth权重统一处理
        if os.path.basename(model_path).lower().endswith('.pth'):
            if torch.cuda.is_available():
                checkpoint = torch.load( model_path)

            else:
                checkpoint = torch.load( model_path, map_location=torch.device("cpu") )
        else:
            checkpoint=None



        # 导入训练好的模型使用，或者继续训练resume
        if keep_last:
            checkpoint_model=checkpoint['state_dict']
            missing_keys, unexpected_keys = net.load_state_dict(checkpoint_model, strict=False)
            # 直接检查哪些层的参数没有被加载  missing_keys: net 里有，但 checkpoint 里没有的参数（例如新的分类层 last_linear）。
            print("Missing keys:", missing_keys)
            # unexpected_keys: checkpoint 里有，但 net 里没有的参数
            print("Unexpected keys:", unexpected_keys)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
        # 前后使用的类别数不同，所以迁移学习中把最后一层权重偏移去掉
        else:
            if model_name == 'efficientformerl3':
                # efficientformerl3
                # 两种情况，一种是用提供的imagenet的预训练权重进行继续训练，一种是用自己训练好的其他数据集的权重继续训练。前者格式得具体设置，后者则是统一state_dict
                try:
                    checkpoint_model = checkpoint['model']
                except:
                    checkpoint_model=checkpoint['state_dict']
                # efficientformerl3
                checkpoint_model.pop("head.weight", None)
                checkpoint_model.pop("head.bias", None)
                # 当调整模型架构（例如移除或替换某些层）时，可以使用 strict=False 以加载与新的模型架构兼容的部分参数
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint_model, strict=False)
                # 直接检查哪些层的参数没有被加载  missing_keys: net 里有，但 checkpoint 里没有的参数（例如新的分类层 last_linear）。
                print("Missing keys:", missing_keys)
                # unexpected_keys: checkpoint 里有，但 net 里没有的参数
                print("Unexpected keys:", unexpected_keys)



                # 统计冻结和未冻结参数数目
                total_params = sum(p.numel() for p in net.parameters())
                trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"Total params: {total_params}")
                print(f"Trainable params: {trainable_params}")
                print(f"Frozen params: {total_params - trainable_params}")

                # 处理 DataParallel 的情况
                model_to_freeze = net.module if isinstance(net, torch.nn.DataParallel) else net
                # efficientformerl3
                # 冻结前面层
                for param in model_to_freeze.parameters():
                    param.requires_grad = False
                # 解冻分类层
                for param in model_to_freeze.head.parameters():
                    param.requires_grad = True

                # 统计冻结和未冻结参数数目
                total_params = sum(p.numel() for p in net.parameters())
                trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

                print(f"Total params: {total_params}")
                print(f"Trainable params: {trainable_params}")
                print(f"Frozen params: {total_params - trainable_params}")


            elif model_name=='inceptionresnetv2':

                # inceptionresnet0
                # checkpoint.pop("linear.weight", None)
                # checkpoint.pop("linear.bias", None)
                # inceptionresnet1
                checkpoint.pop("last_linear.weight", None)
                checkpoint.pop("last_linear.bias", None)
                # 当调整模型架构（例如移除或替换某些层）时，可以使用 strict=False 以加载与新的模型架构兼容的部分参数
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint, strict=False)
                # 直接检查哪些层的参数没有被加载  missing_keys: net 里有，但 checkpoint 里没有的参数（例如新的分类层 last_linear）。
                print("Missing keys:", missing_keys)
                # unexpected_keys: checkpoint 里有，但 net 里没有的参数
                print("Unexpected keys:", unexpected_keys)

                # 处理 DataParallel 的情况
                model_to_freeze = net.module if isinstance(net, torch.nn.DataParallel) else net

                # inceptionresnet1
                # 冻结前面层
                for param in model_to_freeze.parameters():
                    param.requires_grad = False
                # 解冻分类层
                for param in model_to_freeze.last_linear.parameters():
                    param.requires_grad = True
            elif model_name=='customvgg19':
                # custom vgg19
                checkpoint.pop("classifier.0.weight", None)
                checkpoint.pop("classifier.0.bias", None)
                checkpoint.pop("classifier.3.weight", None)
                checkpoint.pop("classifier.3.bias", None)
                checkpoint.pop("classifier.6.weight", None)
                checkpoint.pop("classifier.6.bias", None)

                # 当调整模型架构（例如移除或替换某些层）时，可以使用 strict=False 以加载与新的模型架构兼容的部分参数
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint, strict=False)
                # 直接检查哪些层的参数没有被加载  missing_keys: net 里有，但 checkpoint 里没有的参数（例如新的分类层 last_linear）。
                print("Missing keys:", missing_keys)
                # unexpected_keys: checkpoint 里有，但 net 里没有的参数
                print("Unexpected keys:", unexpected_keys)

                # 处理 DataParallel 的情况
                model_to_freeze = net.module if isinstance(net, torch.nn.DataParallel) else net
                # for i, child in enumerate(model_to_freeze.children()):
                #     print(f"Layer {i}: {child}")
                a=list(model_to_freeze.children())
                for param in model_to_freeze.parameters():
                    param.requires_grad = False
                # 解冻分类层
                for param in model_to_freeze.classifier.parameters():
                    param.requires_grad = True


                # # 打印每一层的参数requires_grad属性 判断是否冻结
                # for name, param in model_to_freeze.named_parameters():
                #     print(f"{name}: {param.requires_grad}")
            # 所有层重新训练
            elif model_name=='customtransunet':
                weights=np.load(model_path)
                net.load_from(weights=weights)

                # for name, param in net.named_parameters():
                #     print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

                # 统计冻结和未冻结参数数目
                total_params = sum(p.numel() for p in net.parameters())
                trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"Total params: {total_params}")
                print(f"Trainable params: {trainable_params}")
                print(f"Frozen params: {total_params - trainable_params}")

                model_to_freeze = net.module if isinstance(net, torch.nn.DataParallel) else net
                # a=list(model_to_freeze.children())

                for param in model_to_freeze.parameters():
                    param.requires_grad = False
                # 解冻分类层
                for param in model_to_freeze.classifier_head.parameters():
                    param.requires_grad = True

                # 统计冻结和未冻结参数数目
                total_params = sum(p.numel() for p in net.parameters())
                trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

                print(f"Total params: {total_params}")
                print(f"Trainable params: {trainable_params}")
                print(f"Frozen params: {total_params - trainable_params}")

            elif model_name=='midfusionmodel':
                # efficientformerl3
                checkpoint = checkpoint['model']
                # efficientformerl3
                checkpoint.pop("head.weight", None)
                checkpoint.pop("head.bias", None)
                # 当调整模型架构（例如移除或替换某些层）时，可以使用 strict=False 以加载与新的模型架构兼容的部分参数
                missing_keys, unexpected_keys = net.efficientformer.load_state_dict(checkpoint, strict=False)
                # 直接检查哪些层的参数没有被加载  missing_keys: net 里有，但 checkpoint 里没有的参数（例如新的分类层 last_linear）。
                print("Missing keys:", missing_keys)
                # unexpected_keys: checkpoint 里有，但 net 里没有的参数
                print("Unexpected keys:", unexpected_keys)



        logger.info(
            "Loaded checkpoint %s in %1.5fs",
            model_path,
            (time.time() - starting_time),
        )
        return checkpoint, net, optimizer, scaler


def save_model(epoch: int, model, loss: float, optimizer, scaler, filename: str):

    model_params = {
        "epoch": epoch,
        "state_dict": copy.deepcopy(model),
        "best_loss": loss,
        "optimizer": copy.deepcopy(optimizer),
        "scaler": scaler,
    }
    torch.save(model_params, filename)
