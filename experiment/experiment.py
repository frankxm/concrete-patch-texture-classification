# -*- coding: utf-8 -*-

import logging
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model
import torch.nn as nn
from evaluate import run as evaluate

from predict import predict_multimodal
from predict import run as predict
from training import run as train
from doc_functions import DLACollateFunction,DLACollateFunction_for_prediction, Sampler,DLACollateFunction_multimodal
from preprocessing import (
    Normalize,
    PredictionDataset,
    ToTensor,
    TrainingDataset,
    apply_augmentations_and_compute_stats,apply_augmentations_and_compute_stats_pred,get_texture,config_dict,
    random_perspective_transform, random_elastic_transform, random_rotate, random_flip,random_gaussian_blur, random_gaussian_noise, random_sharpen, random_contrast)

import json
import pandas as pd
logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import visualize_features
import re


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def training_loaders_multimodal(
    exp_data_paths: dict,
    img_size: int,
    bin_size: int,
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
                Normalize(mean_aug.tolist() if set=='train' else  norm_params['train']['mean'].tolist(), std_aug.tolist() if set=='tain' else norm_params['train']['std'].tolist())
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
                    bin_size=bin_size,
                    batch_size=batch_size if set=="train" else 4,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set=="train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction_multimodal(model_name, mean_features, std_features) ,
                prefetch_factor=prefetch_factor
            )
        else:
            loaders[set] = DataLoader(
                dataset,
                num_workers=0,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    bin_size=bin_size,
                    batch_size=batch_size if set=="train" else 4,
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
    bin_size: int,
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
                Normalize(mean_aug.tolist() if set=='train' else  norm_params['train']['mean'].tolist(), std_aug.tolist() if set=='tain' else norm_params['train']['std'].tolist())
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


        if num_workers > 0:
            loaders[set] = DataLoader(
                dataset,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    bin_size=bin_size,
                    batch_size=batch_size if set=="train" else 4,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set=="train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction(model_name, mean_features, std_features) if model_name in ['texture_model'] else DLACollateFunction(model_name),
                prefetch_factor=prefetch_factor
            )
        else:
            loaders[set] = DataLoader(
                dataset,
                num_workers=0,
                generator=generator,
                pin_memory=True,
                batch_sampler=Sampler(
                    augment_all,
                    bin_size=bin_size,
                    batch_size=batch_size if set == "train" else 4,
                    no_of_epochs=no_of_epochs,
                    israndom=True if set == "train" else False,
                    generator=generator
                ),
                collate_fn=DLACollateFunction(model_name, mean_features, std_features) if model_name in ['texture_model'] else DLACollateFunction(model_name),
            )
        logging.info(f"{set}: Found {len(dataset)} images ")



        if set=='train'and model_name=='machine_learning':

            #####纹理描述符输入
            logging.info(f"{model_name},{set}:Calculting statistical feature descriptors ")
            image = [item[0] for item in augment_all.values()]
            mask = [item[1] for item in augment_all.values()]
            X, names, feature_names = get_texture(image, **config_dict)
            # axis=0对所有样本的同一特征（列）求均值，方差
            mean_features = X.mean(axis=0)  # shape: (532,)
            std_features = X.std(axis=0)
            savepath1 = os.path.join('test', f'mean_featuresnorm{fold}.npy')
            savepath2 = os.path.join('test', f'std_featuresnorm{fold}.npy')
            np.save(savepath1, mean_features)
            np.save(savepath2, std_features)
            X_scaled = (X - mean_features) / std_features
            # 定义分类器
            # Bagging-rf 数据随机采样（Bootstrap）形成多个训练集,	多个模型 并行训练 各子模型“投票”或“平均”输出，多数或平均
            classifiers = {
                'RandomForest':RandomForestClassifier(
                                        n_estimators=400,
                                        max_depth=12,               # 限制树深防止过拟合
                                        min_samples_split=5,        # 内部节点最小划分样本数
                                        min_samples_leaf=3,         # 叶子节点最少样本
                                        max_features='sqrt',        # 每次分裂考虑特征数
                                        class_weight='balanced',    # 处理类别不均衡
                                        random_state=42,
                                        n_jobs=-1
                                                                ),
                'KNN': KNeighborsClassifier(n_neighbors=7,weights='distance'),
                # Boosting每轮训练关注上轮模型表现，调整样本权重  多个模型 串行训练，后一轮依赖前一轮输出 各子模型“加权求和”，迭代学习，累计前面经验
                'XGBOOST':XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,subsample=0.9,colsample_bytree=0.8,random_state=42),
                'LightGBM':LGBMClassifier(n_estimators=300, learning_rate=0.05,num_leaves=63,subsample=0.9,colsample_bytree=0.8, random_state=42)
            }
            # from sklearn.feature_selection import mutual_info_classif,SelectKBest
            # selector = SelectKBest(mutual_info_classif, k=200)
            # X_selected = selector.fit_transform(X_scaled, np.array(mask))
            # 保存 selector
            # selector_path = os.path.join('test', f'selector_fold{fold}.joblib')
            # joblib.dump(selector, selector_path)

            for name, clf in classifiers.items():
                print(f"\nTraining {name}...")

                clf.fit(X_scaled, np.array(mask))
                texture_info['classifier'].append(clf)
                texture_info['name'].append(name)

        elif set=='val' and model_name=='machine_learning':
            logging.info(f"{model_name},{set}:Calculting statistical feature descriptors ")
            image = [item[0] for item in augment_all.values()]
            mask = [item[1] for item in augment_all.values()]
            X, names, feature_names = get_texture(image, **config_dict)
            savepath1 = os.path.join('test', f'mean_featuresnorm{fold}.npy')
            savepath2 = os.path.join('test', f'std_featuresnorm{fold}.npy')
            mean_features = np.load(savepath1)
            std_features = np.load(savepath2)
            X_scaled = (X - mean_features) / std_features

            # selector_path = os.path.join('test', f'selector_fold{fold}.joblib')
            # selector = joblib.load(selector_path)
            # X_selected = selector.transform(X_scaled)

            for clf in texture_info['classifier']:
                index=texture_info['classifier'].index(clf)
                y_probs = clf.predict_proba(X_scaled)
                y_pred = clf.predict(X_scaled)
                acc = accuracy_score(np.array(mask), y_pred)
                name=texture_info['name'][index]
                print(f"{name} Accuracy: {acc:.4f}")

                # 保存模型
                model_path = f"test/{name}_norm{fold}.joblib"
                joblib.dump(clf, model_path)
                print(f"{name} model saved to: {model_path}")





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

def prediction_loaders_multimodal(
         exp_data_paths, img_size,mean_img,std_img,mean_texture,std_texture,model_name
) -> dict:
    loaders = {}
    set='test'
    images= exp_data_paths["test"]["image"]
    augment_all = apply_augmentations_and_compute_stats_pred(images, img_size, set, )

    #测试集的标准化参数必须用训练集的！！！ 未来的新数据是未知分布的样本，不能提前知道它的均值和方差。训练阶段统计好标准化参数，之后所有数据（验证、测试、生产）都必须用相同的转换方式。
    #  训练阶段不用显示转换为tensor是因为有collate_fn（DLACollateFunction）
    dataset = PredictionDataset(
        augment_all,
        transform=transforms.Compose(
            [
                Normalize(mean_img.tolist(), std_img.tolist()),
            ]
        ),model_name=model_name
    )
    loaders[set + "_loader"] = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=DLACollateFunction_for_prediction(model_name,mean_texture,std_texture)
    )


    return loaders


def prediction_loaders(
         exp_data_paths, img_size,mean,std,model_name,log_path,prediction_path,model_path,selector=None
) -> dict:
    p = exp_data_paths["train"]["image"]
    fold = [part for part in p.parts if part.startswith('fold_')][0]

    loaders = {}
    set='test'
    images= exp_data_paths["test"]["image"]
    augment_all = apply_augmentations_and_compute_stats_pred(images, img_size, set)

    #测试集的标准化参数必须用训练集的！！！ 未来的新数据是未知分布的样本，不能提前知道它的均值和方差。训练阶段统计好标准化参数，之后所有数据（验证、测试、生产）都必须用相同的转换方式。
    #  训练阶段不用显示转换为tensor是因为有collate_fn（DLACollateFunction）
    dataset = PredictionDataset(
        augment_all,
        transform=transforms.Compose(
            [
                Normalize(mean.tolist(), std.tolist()),
                ToTensor()
            ]
        ),model_name=model_name
    )
    loaders[set + "_loader"] = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=DLACollateFunction_for_prediction(model_name,mean,std) if model_name=='texture_model' else None
    )

    if model_name == 'machine_learning':


        image = [item for item in augment_all.values()]
        image_names = [item for item in augment_all.keys()]
        X, names, feature_names = get_texture(image, **config_dict)
        X_scaled = (X - mean) / std
        model = joblib.load(model_path)

        # X_selected = selector.transform(X_scaled)

        probs = model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)

        results = []
        for name, pred, prob in zip(image_names, preds, probs):
            results.append({
                "image": name,
                "class": pred,
                "confidence": round(np.max(prob), 4)
            })



        # 保存 CSV
        output_csv_dir = os.path.join(log_path, prediction_path, 'test')
        os.makedirs(output_csv_dir, exist_ok=True)
        output_csv_path = os.path.join(output_csv_dir, "predictions.csv")
        pd.DataFrame(results).to_csv(output_csv_path, index=False)

        print(f"Prediction CSV saved to: {output_csv_path}")




    return loaders

def get_optimizer_with_layerwise_lr(net, lr, weight_decay, backbone_lr_ratio=0.1):
    backbone_decay = []
    backbone_no_decay = []

    other_decay = []
    other_no_decay = []
    #
    # for name, param in net.named_parameters():
    #     print(name)
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        # 判断是否是 backbone（EfficientFormer）
        backbone_params = set(id(p) for p in net.efficientformer.parameters())
        for name, param in net.named_parameters():
            if id(param) in backbone_params:
                is_backbone = True
            else:
                is_backbone = False

        # 判断是否需要 weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or 'norm' in name.lower():
            if is_backbone:
                backbone_no_decay.append(param)
            else:
                other_no_decay.append(param)
        else:
            if is_backbone:
                backbone_decay.append(param)
            else:
                other_decay.append(param)

    param_groups = [
        #backbone（小LR）
        {
            'params': backbone_decay,
            'lr': lr * backbone_lr_ratio,
            'weight_decay': weight_decay
        },
        {
            'params': backbone_no_decay,
            'lr': lr * backbone_lr_ratio,
            'weight_decay': 0.0
        },

        # 其他模块（大LR）
        {
            'params': other_decay,
            'lr': lr,
            'weight_decay': weight_decay
        },
        {
            'params': other_no_decay,
            'lr': lr,
            'weight_decay': 0.0
        },
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


    USE_UNIFORM_WEIGHT = True

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
            "criterion": nn.CrossEntropyLoss(label_smoothing=label_smooth,weight=train_weights_torch),
            "criterion_val": nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=val_weights_torch),
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
            "criterion": nn.CrossEntropyLoss(label_smoothing=label_smooth,weight=train_weights_torch),
            "criterion_val": nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=val_weights_torch),
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


def prediction_initialization(
    model_path: str, classes_names: list,use_gpu,model_name
) -> dict:

    no_of_classes = len(classes_names)
    net = model.load_network(no_of_classes, False,use_gpu,model_name)
    _, net, _, _ = model.restore_model(net, None, None,model_path,model_name)
    return net


def run(config: dict, num_workers: int = 0):
    assert len(config["steps"]) > 0, "No step to run"
    run_experiment(config=config, num_workers=num_workers)


def run_experiment(config: dict, num_workers: int ):

    assert len(config["steps"]) > 0, "No step to run"
    norm_params={"train": {}}

    if "train" in config["steps"]:
        if config["model_name"]=="midfusionmodel":
            loaders, norm_params,proportion = training_loaders_multimodal(
                exp_data_paths=config["data_paths"],
                img_size=config["img_size"],
                bin_size=config["bin_size"],
                batch_size=config["batch_size"],
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
                bin_size=config["bin_size"],
                batch_size=config["batch_size"],
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

        if config['model_name']!='machine_learning':
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
                config["learning_rate"],
                config["same_classes"],
                config["loss"],
                config["use_gpu"],
                config["model_name"],
                config["label_smooth"],
                config["weight_decay"],
                proportion,
                config['log_path']
            )
            train(
                config["model_path"],
                config["log_path"],
                config["tb_path"],
                config["no_of_epochs"],
                norm_params,
                config["classes_names"],
                loaders,
                tr_params,
                config["batch_size"],
                config["desired_batchsize"],
                config["learning_rate"],
                config["use_gpu"],
                config["model_name"]
            )


    if "prediction" in config["steps"]:
        if config['model_name'] == 'machine_learning' or config['model_name'] == 'texture_model':
            mean = np.load(config["mean_features"])
            std = np.load(config["std_features"])
            # selector = joblib.load(config["best_features"])
        else:
            mean, std = load_mean_std(config["norm_params"])
        img_dir = getimg(config["data_paths"]['test']['image'],config["data_paths"]['label_csv']['label'],config["log_path"],config["prediction_path"],config["extra_test_data"])
        if config['model_name'] in ['midfusionmodel']:
            mean_img,std_img=load_mean_std(config["norm_params"])
            mean_texture=np.load(config["mean_features"])
            std_texture=np.load(config["std_features"])
            # selector = joblib.load(config["best_features"])
            loaders=prediction_loaders_multimodal(
                config["data_paths"], config["img_size"],mean_img,std_img,mean_texture,std_texture,config["model_name"]
            )
        else:
            # machine learning情况在这里predict
            loaders= prediction_loaders(
                config["data_paths"], config["img_size"],mean,std,config["model_name"],config["log_path"],config["prediction_path"],config["model_path"]
            )
        if config['model_name'] != 'machine_learning':
            # net = prediction_initialization(
            #     str(config["model_path"]), config["classes_names"],config["use_gpu"], config["model_name"]
            # )
            nets = []
            val_loss_list=[]
            model_paths = config["model_paths"]

            for path in model_paths:
                if path is not None:
                    path_str = str(path)
                    match = re.search(r'loss_([0-9.]+)\.pth$', path_str)
                    if match:
                        val_loss = float(match.group(1))
                    else:
                        val_loss = None
                    val_loss_list.append(val_loss)

                    print(f"{path_str} -> val_loss={val_loss}")

                    net = prediction_initialization(
                        str(path),
                        config["classes_names"],
                        config["use_gpu"],
                        config["model_name"]
                    )
                    nets.append(net)

            best_loss = min(val_loss_list)
            weights = [best_loss / loss for loss in val_loss_list]
            # optional: 转 numpy
            weights = np.array(weights, dtype=np.float32)


            if config['model_name'] in ['midfusionmodel']:
                predict_multimodal(
                    config["prediction_path"],
                    config["log_path"],
                    config["classes_names"],
                    loaders,
                    nets,
                    img_dir,
                    config["use_gpu"],
                    weights,

                )

            else:
                predict(
                    config["prediction_path"],
                    config["log_path"],
                    config["classes_names"],
                    loaders,
                    nets,
                    img_dir,
                    config["use_gpu"],
                    config["visualization"],
                    mean,std,
                    weights,
                )


    if "evaluation" in config["steps"]:
        for set in config["data_paths"].keys():
            if set =='train' or set=='val' or set=='label_csv':
                continue

            logpath=str(config["log_path"])
            predir=os.path.join(logpath,config["prediction_path"],set)
            evaldir=os.path.join(logpath,config["evaluation_path"],set)
            if not os.path.exists(evaldir):
                os.makedirs(evaldir,exist_ok=True)
            if len(os.listdir(predir)) == 0:
                logging.info(f"{predir} folder not found.")
            else:
                logging.info(f"Starting evaluation in {predir}" )
                evaluate(
                    config["classes_names"],
                    config["data_paths"]["label_csv"],
                    evaldir,
                    predir,
                    config["filtered_label_evaluation"]
                )
def getimg(path,label_path,log_path,prediction_path,extra_test_data):
    outputdir=os.path.join(log_path, prediction_path, 'test')
    os.makedirs(outputdir, exist_ok=True)
    csv_output_path = os.path.join(outputdir, "label.csv")
    imgdir={}
    df = pd.read_csv(label_path)
    df.set_index("image", inplace=True)

    test_image_names=[]
    for p in os.listdir(str(path)):
        if p.lower().endswith(('.png', '.jpg', '.jpeg','.tiff')):
            img_bgr=cv2.imread(os.path.join(path,p))
            imgdir[p.split('.')[0]]=img_bgr
            test_image_names.append(p)
    # 如果是用的划分好的测试集，则提前生成测试集的label标签
    if not extra_test_data:
        gt_labels = np.array([df.loc[img, "class"] for img in test_image_names])
        if "class2" in df.columns:
            gt_labels2 = np.array([df.loc[img, "class2"] for img in test_image_names])
        else:
            gt_labels2 = None
        df_gt_labels = pd.DataFrame({
            "image": np.array(test_image_names),
            "class": gt_labels,
            "class2":gt_labels2
        })
        df_gt_labels.to_csv(csv_output_path, index=False)

    return imgdir