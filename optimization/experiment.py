# -*- coding: utf-8 -*-
import json
import logging
import os
import random

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model

from training import run as train
from doc_functions import DLACollateFunction, Sampler,DLACollateFunction_multimodal
from preprocessing import (
    Normalize,
    TrainingDataset,
    apply_augmentations_and_compute_stats,get_texture,config_dict,
    random_perspective_transform, random_elastic_transform, random_rotate, random_flip,random_gaussian_blur, random_gaussian_noise, random_sharpen, random_contrast)

from training_utils import SoftLabelLoss
logger = logging.getLogger(__name__)

import optuna


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def training_loaders_multimodal(
    exp_data_paths: dict,
    img_size: int,
    batch_size: int,
    no_of_epochs:int,
    num_workers: int ,
    norm_params:dict,
    forbid_augmentations:bool,
    model_name:str,
    prefetch_factor,
    use_images_generees,
    images_generees_path,
    log_path,
    classes_names,
    num_genere,
    seed

) -> dict:

    loaders = {}
    proportion={}
    t = tqdm(["train", "val"])
    t.set_description("Loading data")
    generator = torch.Generator()
    generator.manual_seed(seed)

    for set, images in zip( t,[exp_data_paths["train"]["image"], exp_data_paths["val"]["image"]]):
        if set=='train':
            augment_all, mean_aug, std_aug,ratio = apply_augmentations_and_compute_stats(images,img_size,set,exp_data_paths["label_csv"]["label"],use_images_generees,images_generees_path,classes_names,num_genere)
            norm_params[set]['mean']=mean_aug
            norm_params[set]['std']=std_aug
            proportion[set] = ratio

        else:
            augment_all,ratio= apply_augmentations_and_compute_stats(images,img_size,set,exp_data_paths["label_csv"]["label"])
            proportion[set] = ratio
        dataset = TrainingDataset(
            augment_all,
            transform=transforms.Compose([
                Normalize(mean_aug.tolist() if set=='train' else  norm_params['train']['mean'].tolist(), std_aug.tolist() if set=='train' else norm_params['train']['std'].tolist())
            ]),
            augmentations_transformation=[random_perspective_transform, random_elastic_transform, random_rotate, random_flip ] if set=='train' else None,
            augmentations_pixel=[random_gaussian_blur,random_gaussian_noise,random_sharpen,random_contrast] if set=='train' else None,
            forbid=forbid_augmentations,
            model_name=model_name,
            generator=generator
        )

        if set=='train':
            logging.info(f"{model_name},{set}:Calculting statistical feature descriptors ")
            image = [item[0] for item in augment_all.values()]
            X, names, feature_names = get_texture(image, **config_dict)
            mean_features = X.mean(axis=0)  # shape: (532,)
            std_features = X.std(axis=0)
            print('mean_features',mean_features,'std_features',std_features)
            savepath1=os.path.join(log_path,'mean_features.npy')
            savepath2 = os.path.join(log_path, 'std_features.npy')
            np.save(savepath1, mean_features)
            np.save(savepath2, std_features)

            mask = [item[1] for item in augment_all.values()]

            # # # tsne降维可视化，532维->2维 dataset2 augmented
            # visualize_features.tsne(X, np.array(mask),len(classes_names))
            # a=1



        if num_workers > 0:
            loaders[set] = DataLoader(
                dataset,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    batch_size=batch_size if set=="train" else 4,
                    # batch_size=batch_size ,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set=="train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction_multimodal(model_name, mean_features, std_features) ,
                prefetch_factor=prefetch_factor,
            )
        else:
            loaders[set] = DataLoader(
                dataset,
                num_workers=0,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    batch_size=batch_size if set=="train" else 4,
                    # batch_size=batch_size ,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set == "train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction_multimodal(model_name, mean_features, std_features),
            )
        logging.info(f"{set}: Found {len(dataset)} images ")


    return loaders,norm_params,proportion



def training_loaders(
    exp_data_paths: dict,
    img_size: int,
    batch_size: int,
    no_of_epochs:int,
    num_workers: int ,
    norm_params:dict,
    forbid_augmentations:bool,
    model_name:str,
    prefetch_factor,
    use_images_generees,
    images_generees_path,
    log_path,
    classes_names,
    num_genere,
    seed

) -> dict:

    p = exp_data_paths["train"]["image"]
    fold = [part for part in p.parts if part.startswith('fold_')][0]


    loaders = {}
    proportion={}
    t = tqdm(["train", "val"])
    t.set_description("Loading data")
    texture_info={'classifier':[],'name':[]}
    generator = torch.Generator()
    generator.manual_seed(seed)
    for set, images in zip( t,[exp_data_paths["train"]["image"], exp_data_paths["val"]["image"]]):
        if set=='train':
            augment_all, mean_aug, std_aug ,ratio= apply_augmentations_and_compute_stats(images,img_size,set,exp_data_paths["label_csv"]["label"],use_images_generees,images_generees_path,classes_names,num_genere)
            norm_params[set]['mean']=mean_aug
            norm_params[set]['std']=std_aug
            proportion[set]=ratio



        else:
            augment_all,ratio= apply_augmentations_and_compute_stats(images,img_size,set,exp_data_paths["label_csv"]["label"])
            proportion[set] = ratio
        dataset = TrainingDataset(
            augment_all,
            transform=transforms.Compose([
                Normalize(mean_aug.tolist() if set=='train' else  norm_params['train']['mean'].tolist(), std_aug.tolist() if set=='train' else norm_params['train']['std'].tolist())
            ]),
            augmentations_transformation=[random_perspective_transform, random_elastic_transform, random_rotate, random_flip ] if set=='train' else None,
            augmentations_pixel=[random_gaussian_blur,random_gaussian_noise,random_sharpen,random_contrast] if set=='train' else None,
            forbid=forbid_augmentations,
            model_name=model_name,
            generator=generator
        )

        if set=='train'and model_name=='texture_model' :
            logging.info(f"{model_name},{set}:Calculting statistical feature descriptors ")
            image = [item[0] for item in augment_all.values()]
            X, names, feature_names = get_texture(image, **config_dict)
            mean_features = X.mean(axis=0)  # shape: (532,)
            std_features = X.std(axis=0)
            print('mean_features',mean_features,'std_features',std_features)
            savepath1=os.path.join(log_path,f'mean_features_{fold}.npy')
            savepath2 = os.path.join(log_path, f'std_features_{fold}.npy')
            np.save(savepath1, mean_features)
            np.save(savepath2, std_features)

#val batchsize 指定为4，防止val过小
        if num_workers > 0:
            loaders[set] = DataLoader(
                dataset,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    batch_size=batch_size if set=="train" else 4,
                    # batch_size=batch_size ,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set=="train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction(model_name, mean_features, std_features) if model_name in ['texture_model'] else DLACollateFunction(model_name),
                prefetch_factor=prefetch_factor,
            )
        else:
            loaders[set] = DataLoader(
                dataset,
                num_workers=0,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    batch_size=batch_size if set == "train" else 4,
                    # batch_size=batch_size ,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set == "train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction(model_name, mean_features, std_features) if model_name in ['texture_model'] else DLACollateFunction(model_name),
            )
        logging.info(f"{set}: Found {len(dataset)} images ")



    return loaders,norm_params,proportion


def load_mean_std(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    mean_line = next(line for line in lines if line.startswith('mean:'))
    std_line = next(line for line in lines if line.startswith('std:'))

    # 提取方括号内的内容
    mean_str = mean_line.split('[')[1].split(']')[0]
    std_str = std_line.split('[')[1].split(']')[0]


    # 转换成 numpy 数组
    mean = np.fromstring(mean_str, sep=' ')
    std = np.fromstring(std_str, sep=' ')

    return mean, std

def get_optimizer_with_layerwise_lr_midfusion(
    net,
    lr,
    weight_decay,
    backbone_lr_ratio=0.1,
):
    backbone_decay = []
    backbone_no_decay = []

    other_decay = []
    other_no_decay = []

    # ========================================================
    # EfficientFormer 的所有参数身份
    # ========================================================
    backbone_params = set(
        id(p) for p in net.efficientformer.parameters()
    )

    # ========================================================
    # 遍历所有 trainable parameters
    # ========================================================
    for name, param in net.named_parameters():

        if not param.requires_grad:
            continue

        # 判断是否属于 EfficientFormer
        is_backbone = id(param) in backbone_params

        # 判断是否需要 weight decay
        no_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )

        if is_backbone:

            if no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        else:

            if no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)

    # ========================================================
    # Parameter groups
    # ========================================================
    param_groups = [
        # 其他 MidFusion 模块：大 LR
        {
            "params": other_decay,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": other_no_decay,
            "lr": lr,
            "weight_decay": 0.0,
        },
        # EfficientFormer：小 LR
        {
            "params": backbone_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": weight_decay,
        },
        {
            "params": backbone_no_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": 0.0,
        },


    ]

    # 删除空 parameter groups
    param_groups = [
        group for group in param_groups
        if len(group["params"]) > 0
    ]

    return AdamW(param_groups)

def get_optimizer_with_layerwise_lr_inception(
    net,
    lr,
    weight_decay,
    backbone_lr_ratio=0.01,
    conv7b_lr_ratio=0.1,
):
    backbone_decay = []
    backbone_no_decay = []

    conv7b_decay = []
    conv7b_no_decay = []

    head_decay = []
    head_no_decay = []

    # ========================================================
    # 参数身份集合
    # ========================================================

    # 最后的特征投影层
    conv7b_params = set(
        id(p) for p in net.conv2d_7b.parameters()
    )

    # 分类头
    head_params = set(
        id(p) for p in net.last_linear.parameters()
    )

    # ========================================================
    # 遍历所有 trainable parameters
    # ========================================================

    for name, param in net.named_parameters():

        if not param.requires_grad:
            continue

        param_id = id(param)

        # 判断是否不使用 weight decay
        no_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        )

        # ----------------------------------------------------
        # 分类头：最高 LR
        # ----------------------------------------------------
        if param_id in head_params:

            if no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        # ----------------------------------------------------
        # conv2d_7b：中等 LR
        # ----------------------------------------------------
        elif param_id in conv7b_params:

            if no_decay:
                conv7b_no_decay.append(param)
            else:
                conv7b_decay.append(param)

        # ----------------------------------------------------
        # 其他 backbone：较低 LR
        # ----------------------------------------------------
        else:

            if no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    # ========================================================
    # Parameter groups
    # ========================================================

    param_groups = [

        # Backbone：较低 LR
        {
            "params": backbone_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": weight_decay,
        },
        {
            "params": backbone_no_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": 0.0,
        },

        # conv2d_7b：中等 LR
        {
            "params": conv7b_decay,
            "lr": lr * conv7b_lr_ratio,
            "weight_decay": weight_decay,
        },
        {
            "params": conv7b_no_decay,
            "lr": lr * conv7b_lr_ratio,
            "weight_decay": 0.0,
        },

        # last_linear：较高 LR
        {
            "params": head_decay,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": head_no_decay,
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]

    # 删除空 parameter groups
    param_groups = [
        group for group in param_groups
        if len(group["params"]) > 0
    ]

    return AdamW(param_groups)
def get_optimizer_with_layerwise_lr(
    net,
    lr,
    weight_decay,
    backbone_lr_ratio=0.1
):
    stage4_decay = []
    stage4_no_decay = []

    other_decay = []
    other_no_decay = []

    # ========================================================
    # 只把 EfficientFormer 的 Stage4 定义为小 LR 部分
    # ========================================================
    stage4_params = set(
        id(p) for p in net.efficientformer.network[6].parameters()
    )

    # ========================================================
    # 遍历所有 trainable parameters
    # ========================================================
    for name, param in net.named_parameters():

        if not param.requires_grad:
            continue

        # 判断是否属于 Stage4
        is_stage4 = id(param) in stage4_params

        # ====================================================
        # 判断是否需要 weight decay
        # ====================================================
        no_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )

        if is_stage4:

            if no_decay:
                stage4_no_decay.append(param)
            else:
                stage4_decay.append(param)

        else:

            if no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)

    # ========================================================
    # Parameter groups
    # ========================================================
    param_groups = [

        # ----------------------------------------------------
        # Stage4：小 LR
        # ----------------------------------------------------
        {
            "params": stage4_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": weight_decay,
        },
        {
            "params": stage4_no_decay,
            "lr": lr * backbone_lr_ratio,
            "weight_decay": 0.0,
        },

        # ----------------------------------------------------
        # Head + Norm：大 LR
        # ----------------------------------------------------
        {
            "params": other_decay,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": other_no_decay,
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]

    # 删除空 parameter groups
    param_groups = [
        group for group in param_groups
        if len(group["params"]) > 0
    ]

    return AdamW(param_groups)

def get_optimizer_with_weight_decay(net, lr, weight_decay):
    decay_params = []
    no_decay_params = []
    decay_params_names = []
    no_decay_params_names = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or 'norm' in name.lower():
            no_decay_params.append(param)
            no_decay_params_names.append(name)
        else:
            decay_params.append(param)
            decay_params_names.append(name)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    return AdamW(param_groups, lr=lr)

def training_initialization(
    training: str,
    classes_names: list,
    use_amp: bool,
    learning_rate: float,
    same_classes:bool,
    loss:str,
    use_gpu:bool,
    model_name:str,
    label_smooth,
    weight_decay,
    proportion,
    logpath
) -> dict:


    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    epsilon=1e-6


    #计算 train 权重
    train_prop = np.array([v[1] for v in proportion['train'].values()])
    inv_train_raw = 1.0 / (train_prop + epsilon)
    train_weights_before = torch.tensor(inv_train_raw, dtype=torch.float32)
    # 归一化
    inv_train_norm = inv_train_raw / np.mean(inv_train_raw)
    train_weights_norm = torch.tensor(inv_train_norm, dtype=torch.float32)

    val_prop = np.array([v[1] for v in proportion['val'].values()])
    inv_val_raw = 1.0 / (val_prop + epsilon)
    val_weights_before = torch.tensor(inv_val_raw, dtype=torch.float32)
    inv_val_norm = inv_val_raw / np.mean(inv_val_raw)
    val_weights_norm = torch.tensor(inv_val_norm, dtype=torch.float32)


    USE_UNIFORM_WEIGHT = False

    if USE_UNIFORM_WEIGHT:
        train_weights_torch = torch.ones_like(train_weights_norm).to(device)
        val_weights_torch = torch.ones_like(val_weights_norm).to(device)
    else:
        train_weights_torch = train_weights_norm.to(device)
        val_weights_torch = val_weights_norm.to(device)



    # ===== 日志记录 =====
    savepath = os.path.join(logpath, 'weight_loss.txt')

    with open(savepath, "a") as file:
        file.write("\n==============================\n")

        # 真实使用的权重
        used_train_str = [f"{v:.4f}" for v in train_weights_torch.cpu().tolist()]
        used_val_str = [f"{v:.4f}" for v in val_weights_torch.cpu().tolist()]

        file.write("weight_used(train): " + str(used_train_str) + "\n")
        file.write("weight_used(val): " + str(used_val_str) + "\n")

        # 归一化权重（理论值）
        norm_train_str = [f"{v:.4f}" for v in train_weights_norm.tolist()]
        norm_val_str = [f"{v:.4f}" for v in val_weights_norm.tolist()]

        file.write("weight_normed(train): " + str(norm_train_str) + "\n")
        file.write("weight_normed(val): " + str(norm_val_str) + "\n")

        # 3原始反比例权重
        raw_train_str = [f"{v:.4f}" for v in train_weights_before.tolist()]
        raw_val_str = [f"{v:.4f}" for v in val_weights_before.tolist()]

        file.write("weight_raw_inverse(train): " + str(raw_train_str) + "\n")
        file.write("weight_raw_inverse(val): " + str(raw_val_str) + "\n")


    no_of_classes = len(classes_names)
    net = model.load_network(no_of_classes, use_amp,use_gpu,model_name)
    net.apply(model.weights_init)
    if training is None:
        tr_params = {
            "net": net,
            # "criterion": nn.CrossEntropyLoss(label_smoothing=label_smooth,weight=train_weights_torch),
            # "criterion_val": nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=val_weights_torch),
            "criterion": SoftLabelLoss(num_classes=len(classes_names), weights=train_weights_torch,label_smooth=label_smooth),
            "criterion_val": SoftLabelLoss(num_classes=len(classes_names), weights=val_weights_torch,label_smooth=label_smooth),
            # L2 正则化（也叫 Ridge Regularization）可以通过在 损失函数中增加一个正则项 来抑制权重过大，从而提高泛化能力。
            # 小数据集（容易过拟合）：尝试 0.001 或更大  大数据集（不容易过拟合）：尝试 0.0001  如果训练 loss 降低但测试 loss 升高，说明过拟合，可以增加 weight_decay  如果 loss 下降太慢，可能 weight_decay 太大了，可以减少它
            # Adam+weight_decay会失效 L2正则和Weight Decay在Adam这种自适应学习率算法中并不等价，只有在标准SGD的情况下，可以将L2正则和Weight Decay看做一样。
            # "optimizer": AdamW(net.parameters(), lr=learning_rate,weight_decay=weight_decay),
            #  LayerNorm、bias、BatchNorm 等参数上，它们是不应该有 weight decay
            "optimizer": get_optimizer_with_weight_decay(net, lr=learning_rate, weight_decay=weight_decay),

            "saved_epoch": 0,
            "best_loss": 10e5,
            "scaler": GradScaler(enabled=use_amp),
            "use_amp": use_amp,
        }
        logger.info(f"Initialize model: {model_name} successfully")
    else:
        # Restore model to resume training.
        checkpoint, net, optimizer, scaler = model.restore_model(
            net,
            get_optimizer_with_weight_decay(net, lr=learning_rate, weight_decay=weight_decay),
            GradScaler(enabled=use_amp),
            str(training),
            model_name,
            same_classes,

        )
        tr_params = {
            "net": net,
            # "criterion": nn.CrossEntropyLoss(label_smoothing=label_smooth,weight=train_weights_torch),
            # "criterion_val": nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=val_weights_torch),
            "criterion": SoftLabelLoss(num_classes=len(classes_names), weights=train_weights_torch,label_smooth=label_smooth),
            "criterion_val": SoftLabelLoss(num_classes=len(classes_names), weights=val_weights_torch,label_smooth=label_smooth),
            "optimizer": optimizer,
            "best_loss": checkpoint["best_loss"]
            if loss == "best" and checkpoint is not None
            else 10e5,
            "scaler": scaler,
            "use_amp": use_amp,
            "saved_epoch": checkpoint["epoch"]
            if checkpoint is not None and checkpoint.get("epoch", None)
            else 0
        }
    # 例:检测efficientformer冻结情况
    # backbone_ids = set(id(p) for p in net.efficientformer.parameters())
    #
    # backbone_cnt = 0
    # other_cnt = 0
    #
    # for name, param in net.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #     if id(param) in backbone_ids:
    #         backbone_cnt += param.numel()
    #     else:
    #         other_cnt += param.numel()
    #
    # print(f"Backbone params: {backbone_cnt}")
    # print(f"Other params: {other_cnt}")
    # print(f"Total params: {backbone_cnt + other_cnt}")

    return tr_params



def suggest_hyperparameters(trial):

    config = {

        # AdamW learning rate
        "lr": trial.suggest_float(
            "lr",
            1e-6,
            1e-2,
            log=True
        ),

        # AdamW weight decay
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-2,
            log=True
        ),

        # Batch size
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [8,16, 32,64]
        ),

        # Label smoothing
        "label_smoothing": trial.suggest_float(
            "label_smoothing",
            0.1,
            0.4,
            step=0.05
        ),

        # ReduceLROnPlateau patience
        "scheduler_patience": trial.suggest_int(
            "scheduler_patience",
            10,
            30,
            step=5
        ),
    }

    return config
def objective(trial,config,num_workers):

    # --------------------------------------------------------
    # Optuna generates one configuration for this trial
    # --------------------------------------------------------

    optuna_params = suggest_hyperparameters(trial)

    print(
        f"\nTrial {trial.number}"
        f"\nParameters: {optuna_params}"
    )

    result=run_experiment(config=config, num_workers=num_workers,optuna_params=optuna_params,trial=trial)
    trial_dir = os.path.join(
        config["log_path"],
        config["optuna_path"],
        "trials"
    )

    os.makedirs(trial_dir, exist_ok=True)


    trial_result = {
        "trial_number": trial.number,

        # Optuna hyperparameters
        "params": trial.params,

        # Training results
        "best_macro_f1": result["best_macro_f1"],
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
    }

    with open(
            os.path.join(
                trial_dir,
                f"trial_{trial.number}.json"
            ),
            "w"
    ) as f:
        json.dump(
            trial_result,
            f,
            indent=4
        )

    return result["best_macro_f1"]
def create_optuna_study(optuna_dir,fold,seed):

    storage_path = os.path.join(
        optuna_dir,
        f"{fold}.db"
    )

    storage_name = (
        f"sqlite:///{storage_path}"
    )

    study = optuna.create_study(

        study_name=f"dl_{fold}",

        storage=storage_name,

        load_if_exists=True,

        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=seed
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=5,
        ),
    )

    return study
def run(config: dict, num_workers: int = 0):
    assert len(config["steps"]) > 0, "No step to run"

    optuna_dir = (os.path.join(config["log_path"], config["optuna_path"]))
    os.makedirs(optuna_dir, exist_ok=True)
    p = config['data_paths']["train"]["image"]
    fold = [part for part in p.parts if part.startswith('fold_')][0]
    study = create_optuna_study(optuna_dir,fold,config["seed"])

    # ========================================================
    # 3. Hyperparameter optimization
    # ========================================================

    study.optimize( lambda trial: objective(
        trial,
        config,num_workers
    ),n_trials=config["trials"],)

    # ========================================================
    # 4. Best hyperparameters
    # ========================================================

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    print("\nBest validation Macro-F1:")
    print(best_value)

    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    trial_dir = os.path.join(optuna_dir,"trials")
    best_trial_path = os.path.join( trial_dir,f"trial_{best_trial.number}.json" )



    best_result = {
        "best_value": best_value,
        "best_params": best_params,
    }
    best_result_path = os.path.join(
        optuna_dir,
        f"{fold}_best_result.json"
    )

    with open(best_result_path, "w") as f:
        json.dump(
            best_result,
            f,
            indent=4
        )

    print(
        f"\nBest result saved to:\n"
        f"{best_result_path}"
    )







def run_experiment(config: dict, num_workers: int ,optuna_params:dict,trial):

    assert len(config["steps"]) > 0, "No step to run"
    norm_params={"train": {}}

    if "train" in config["steps"]:
        if config["model_name"]=="midfusionmodel":
            loaders, norm_params,proportion = training_loaders_multimodal(
                exp_data_paths=config["data_paths"],
                img_size=config["img_size"],
                batch_size=optuna_params["batch_size"],
                no_of_epochs=config["no_of_epochs"],
                num_workers=num_workers,
                norm_params=norm_params,
                forbid_augmentations=config["forbid_augmentations"],
                model_name=config["model_name"],
                prefetch_factor=config["prefetch_factor"],
                use_images_generees=config['use_images_generees'],
                images_generees_path=config['images_generees_path'],
                log_path=config['log_path'],
                classes_names=config["classes_names"],
                num_genere=config["num_genere"],
                seed=config["seed"]

            )
        else:
            loaders,norm_params,proportion = training_loaders(
                    exp_data_paths=config["data_paths"],
                    img_size=config["img_size"],
                    batch_size=optuna_params["batch_size"],
                    no_of_epochs=config["no_of_epochs"],
                    num_workers=num_workers,
                    norm_params=norm_params,
                    forbid_augmentations=config["forbid_augmentations"],
                    model_name=config["model_name"],
                    prefetch_factor=config["prefetch_factor"],
                    use_images_generees=config['use_images_generees'],
                    images_generees_path=config['images_generees_path'],
                    log_path=config['log_path'],
                    classes_names=config["classes_names"],
                    num_genere=config["num_genere"],
                    seed=config["seed"]

                )

        savepath = os.path.join(config['log_path'], 'norm_params.txt')
        with open(savepath, "w") as file:
            for key, value in norm_params.items():
                file.write(f"set:{key}:" + "\n")
                for k, v in value.items():
                    file.write(f"{k}:" + str(v) + "\n")

        savepath = os.path.join(config['log_path'], 'proportion.txt')
        with open(savepath, "w") as file:
            for key, value in proportion.items():
                file.write(f"set:{key}:" + "\n")
                for val,(count, ratio) in value.items():
                    file.write(f"label: {val}: count={count},ratio={ratio:.4f}\n")

        tr_params = training_initialization(
            config["model_path"],
            config["classes_names"],
            config["use_amp"],
            optuna_params["lr"],
            config["same_classes"],
            config["loss"],
            config["use_gpu"],
            config["model_name"],
            optuna_params["label_smoothing"],
            optuna_params["weight_decay"],
            proportion,
            config['log_path']
        )
        result=train(
            config["log_path"],
            config["tb_path"],
            config["no_of_epochs"],
            config["classes_names"],
            loaders,
            tr_params,
            optuna_params["batch_size"],
            optuna_params["batch_size"],
            config["use_gpu"],
            config["model_name"],
            optuna_params["scheduler_patience"],
            trial
        )
        return result




