# -*- coding: utf-8 -*-

"""
    The train module
    ======================

    Use it to train a model.
"""
import gc
import logging
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import training_utils as tr_utils
import training_pixel_metrics as p_metrics
import model
import math
import torch.optim
from pathlib import Path

def init_metrics(no_of_classes: int) -> dict:

    return {"matrix": np.zeros((no_of_classes, no_of_classes)), "loss": 0}


def log_metrics(epoch: int, metrics: dict, writer,learning_rate, step: str):


    prefixed_metrics = {step + "_" + key: value for key, value in metrics.items()}
    step_name = "TRAIN" if step == "Training" else "VALID"
    for tag, scalar in prefixed_metrics.items():
        writer.add_scalar(tag, scalar, epoch)
        logging.info("  {} {}: {}={}".format(step_name, epoch, tag, round(scalar, 4)))


    if step=='Training':
        # 记录学习率
        writer.add_scalar(f"{step_name}_learning_rate", learning_rate, epoch)
        logging.info("  {} {}: learning_rate={}".format(step_name, epoch, round(learning_rate, 6)))



def run_one_epoch(
    loader,
    params: dict,
    writer,
    epochs: list,
    no_of_epochs: int,
    device: str,
    norm_params: dict,
    classes_names: list,
    batchsize:int,
    desired_batch_size:int,
    logpath,
    model_name,
    step: str,



):
    # 一轮的全局指标
    metrics = init_metrics(len(classes_names))
    epoch = epochs[0]

    total_steps = len(loader)
    t = tqdm(loader, total=total_steps)
    step_name = "TRAIN" if step == "Training" else "VALID"
    t.set_description(
        "{} (prog) {}/{} batchsize:{}".format(step_name, epoch, no_of_epochs + epochs[1], batchsize)
    )

    accumulation_steps = desired_batch_size // batchsize
    # 梯度裁剪的最大范数
    max_grad_norm = 1.0
    # 初始化累积步数计数器
    accumulation_counter = 0

    for index, data in enumerate(t, 1):


        with autocast(enabled=params["use_amp"]):
            torch.cuda.empty_cache()
            if params["use_amp"]:
                if model_name=='midfusionmodel':
                    output = params["net"](data["image"].to(device).half(),data["texture"].to(device).half())
                else:
                    output = params["net"](data["image"].to(device).half())
            else:
                if model_name=='midfusionmodel':
                    output = params["net"](data["image"].to(device).float(),data["texture"].to(device).float())
                else:
                    output = params["net"](data["image"].to(device).float())


            _, preds = torch.max(output, 1)
            loss =  params["criterion"](output, data["label"].to(device))


        original_loss = loss
        if step =="Training":
            # 梯度累计：将损失缩小 当前 mini-batch 的 loss 除以累计步数，这是为了模拟在每个大 batch 下的平均梯度更新。
            loss = loss / accumulation_steps
            params["scaler"].scale(loss).backward()
            accumulation_counter += 1

        # 计算和更新指标,更新指标以原batchsize为准
        for pred in range(output.shape[0]):
            current_pred = output[pred, :].argmax().item()
            # 使用 .item() 转换为标量
            current_label = data["label"][pred].item()

            batch_metrics = p_metrics.compute_metrics(
                current_pred, current_label, original_loss.item(), classes_names
            )
            metrics = p_metrics.update_metrics(metrics, batch_metrics, pred)

        # 如果达到累计步数，执行梯度更新
        if accumulation_counter % accumulation_steps == 0:
            if step == "Training":
                # 梯度裁剪
                params["scaler"].unscale_(params["optimizer"])
                torch.nn.utils.clip_grad_norm_(params["net"].parameters(), max_grad_norm)
                # 更新参数
                params["scaler"].step(params["optimizer"])
                params["scaler"].update()
                # 清零梯度
                params["optimizer"].zero_grad()
                logging.info(f"Terminer les {accumulation_counter} tours de gradient d'accumulation")

            # 重置累积步数计数器
            accumulation_counter = 0
        # 计算到目前批次为止的指标以及loss 一轮的loss整体上是所有批次loss相加除以批次数 对于第n个批次来说loss就是loss_sum/n
        epoch_values = tr_utils.get_epoch_values(metrics, classes_names, index)
        display_values = epoch_values
        display_values["loss"] = round(display_values["loss"], 4)
        t.set_postfix(values=str(display_values))


    # 最后一次梯度更新，防止没有被 accumulation_steps 整除的情况
    if accumulation_counter != 0:
        if step == "Training":
            # 梯度裁剪
            params["scaler"].unscale_(params["optimizer"])
            torch.nn.utils.clip_grad_norm_(params["net"].parameters(), max_grad_norm)
            # 更新参数
            params["scaler"].step(params["optimizer"])
            params["scaler"].update()
            params["optimizer"].zero_grad()
            logging.info(f"Terminer les {accumulation_counter} tours restants de gradient d'accumulation")

    if step == "Training":
        return params, epoch_values
    else:
        return epoch_values
#   模型微调情况下初始学习率大概在1e-3 ~  1e-5
def find_lr(train_dataloader,tr_params,device,batchsize,init_value = 1e-4, final_value=1, beta = 0.98):
    optimizer=tr_params["optimizer"]

    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = float('inf')
    batch_num = 0
    losses = []
    original_losses=[]
    log_lrs = []
    total_steps = math.ceil(len(train_dataloader) / batchsize )
    t = tqdm(train_dataloader, total=total_steps)
    mult = (final_value / init_value) ** (1.0 / total_steps)
    tr_params["net"].train()
    for data in t:
        batch_num += 1

        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        output = tr_params["net"](data["image"].to(device).float())
        loss = tr_params["criterion"](output, data["label"].to(device))
        #通过指数加权平均计算平滑后的损失值 smoothed_loss，使得损失曲线更平滑，减少噪声的影响
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        original_loss=loss.item()

        #如果当前平滑损失突然增长超过之前最低损失的4倍，则停止学习率搜索，认为损失发生了爆炸，返回当前的学习率和损失记录。
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses,original_losses

        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        original_losses.append(original_loss)
        loss.backward()
        optimizer.step()

        lr *= mult

        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses,original_losses

# alltrain
# def run(
#     model_path: str,
#     log_path: str,
#     tb_path: str,
#     no_of_epochs: int,
#     norm_params: dict,
#     classes_names: list,
#     loaders: dict,
#     tr_params: dict,
#     batchsize:int,
#     desired_batchsize:int,
#     learning_rate,
#     use_gpu,
#     model_name
# ):
#     if use_gpu:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device("cpu")
#
#     plateau_scheduler = ReduceLROnPlateau(
#         tr_params["optimizer"],
#         mode='min',
#         factor=0.1,
#         patience=25,
#         verbose=True,
#         min_lr=1e-7,
#     )
#
#
#     writer = SummaryWriter(os.path.join(log_path, tb_path))
#     logging.info("Starting training")
#     starting_time = time.time()
#
#
#
#
#     #  创建optimizer时把model.parameters() 里的 所有参数 放进了一个默认的 param group。
#     for i, group in enumerate(tr_params["optimizer"].param_groups):
#         print(f"[Group {i}] lr = {group['lr']}")
#
#     for epoch in range(1, no_of_epochs + 1):
#
#         current_epoch = epoch + tr_params["saved_epoch"]
#         # Run training.
#         tr_params["net"].train()
#         tr_params, epoch_values = run_one_epoch(
#             loaders["train"],
#             tr_params,
#             writer,
#             [current_epoch, tr_params["saved_epoch"]],
#             no_of_epochs,
#             device,
#             norm_params["train"],
#             classes_names,
#             batchsize,
#             desired_batchsize,
#             log_path,
#             step="Training",
#
#         )
#
#         log_metrics(
#             epoch=current_epoch,
#             metrics=epoch_values,
#             writer=writer,
#             learning_rate=learning_rate,
#             step="Training",
#         )
#
#         # 尝试：学习率固定衰减
#         #  lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#
#         # 更新学习率调度器
#         plateau_scheduler.step(epoch_values["loss"])
#
#         # # 更新学习率
#         # scheduler.step()
#         learning_rate = tr_params["optimizer"].param_groups[0]['lr']
#         logging.info(f"Learning rate now is {learning_rate}")
#
#         # if learning_rate<=0.0001:
#         #     controller.unfreez_all()
#
#         # # 早停策略
#         # early_stopping(epoch_values["loss"])
#         # if early_stopping.stop_training:
#         #     logging.info(f"Early stopping at epoch {current_epoch}")
#         #     print(f"Early stopping at epoch {current_epoch}")
#         #     break
#
#         if model_path is not None:
#             model_last_path = (log_path / model_path).with_name("model_last.pth")
#             model_best_path = (log_path / model_path).with_name("model_best.pth")
#         else:
#             model_last_path = log_path / "texture_model/model_last.pth"
#             model_best_path = log_path / "texture_model/model_best.pth"
#
#         if not os.path.exists(model_last_path.parent):
#             os.makedirs(model_last_path.parent)
#
#         # 每一轮都保存当前模型，覆盖 model_last.pth
#         model.save_model(
#             current_epoch + 1,
#             tr_params["net"].state_dict(),
#             epoch_values["loss"],
#             tr_params["optimizer"].state_dict(),
#             tr_params["scaler"].state_dict(),
#             model_last_path,
#         )
#         logging.info("Latest model (epoch %d) saved to %s", current_epoch + 1, model_last_path)
#
#         # 判断是否为最佳模型，保存为 model_best.pth
#         if epoch_values["loss"] < tr_params["best_loss"]:
#             tr_params["best_loss"] = epoch_values["loss"]
#
#             model.save_model(
#                 current_epoch + 1,
#                 tr_params["net"].state_dict(),
#                 epoch_values["loss"],
#                 tr_params["optimizer"].state_dict(),
#                 tr_params["scaler"].state_dict(),
#                 model_best_path,
#             )
#             logging.info("Best model (epoch %d) saved to %s", current_epoch + 1, model_best_path)
#
#
#
#
#     end = time.gmtime(time.time() - starting_time)
#     logging.info(
#         "Finished training in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
#     )


def run(
    model_path: str,
    log_path: str,
    tb_path: str,
    no_of_epochs: int,
    norm_params: dict,
    classes_names: list,
    loaders: dict,
    tr_params: dict,
    batchsize:int,
    desired_batchsize:int,
    learning_rate,
    use_gpu,
    model_name
):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # early_stopping = EarlyStopping(patience=100, delta=0.01)
    # # Warmup scheduler: 将学习率逐步从接近 0 增加到目标学习率，持续  个 epoch 以避免在训练初期学习率过高导致的不稳定性。
    # warmup_epochs = 20
    # warmup_scheduler = LambdaLR(
    #     tr_params["optimizer"],
    #     lambda epoch: (epoch - 1) / warmup_epochs if epoch <= warmup_epochs else 1.0
    # )
    #
    # # 余弦退火
    # cosine_scheduler = CosineAnnealingLR(
    #     tr_params["optimizer"], T_max=no_of_epochs/2 - warmup_epochs, eta_min=1e-6
    # )
    #
    # # 组合调度器：SequentialLR 先运行 warmup，然后运行余弦退火
    # scheduler = SequentialLR(
    #     tr_params["optimizer"],
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs]
    # )

    # 创建 ReduceLROnPlateau 调度器
    # 我们希望验证集损失最小化
    # 当触发时，学习率乘以 0.1
    # 在日志中打印学习率减少的信息
    # 学习率的下限
    # 相对变化的阈值，1%
    plateau_scheduler = ReduceLROnPlateau(
        tr_params["optimizer"],
        mode='min',
        factor=0.1,
        patience=25,
        verbose=True,
        min_lr=1e-7,
    )


    writer = SummaryWriter(os.path.join(log_path, tb_path))
    logging.info("Starting training")
    starting_time = time.time()



    # logs, losses,original_losses = find_lr(loaders["train"],tr_params,device,batchsize)
    # min_idx = losses.index(min(losses))
    # print(f"最小loss出现在log10(lr)={logs[min_idx]}, lr={10 ** logs[min_idx]}")
    #
    # plt.figure(1, figsize=(20, 10))
    # plt.plot(logs, losses)
    # plt.xlabel('log10(lr) lr')
    # plt.ylabel('smoothloss')
    # plt.title(f'batchsize:{batchsize} loss change avec log10(lr)')
    #
    # plt.savefig(f'{batchsize}-smoothloss.png')
    # plt.figure(2, figsize=(20, 10))
    # plt.plot(logs, original_losses)
    # plt.xlabel('log10(lr) lr')
    # plt.ylabel('originalloss')
    # plt.title(f'batchsize:{batchsize} loss change avec log10(lr)')
    # plt.savefig(f'{batchsize}-originalloss.png')
    # # plt.show()

    # controller = TrainingPhaseController(tr_params["net"])


    #  创建optimizer时把model.parameters() 里的 所有参数 放进了一个默认的 param group。
    for i, group in enumerate(tr_params["optimizer"].param_groups):
        print(f"[Group {i}] lr = {group['lr']}")

    for epoch in range(1, no_of_epochs + 1):
        # if epoch==1:
        #     # 第一轮触发学习率预热
        #     scheduler.step()
        #     learning_rate = tr_params["optimizer"].param_groups[0]['lr']
        #     logging.info(f"Learning rate warm up , is {learning_rate} now")
        current_epoch = epoch + tr_params["saved_epoch"]
        # Run training.
        tr_params["net"].train()
        tr_params, epoch_values = run_one_epoch(
            loaders["train"],
            tr_params,
            writer,
            [current_epoch, tr_params["saved_epoch"]],
            no_of_epochs,
            device,
            norm_params["train"],
            classes_names,
            batchsize,
            desired_batchsize,
            log_path,
            model_name,
            step="Training",

        )

        log_metrics(
            epoch=current_epoch,
            metrics=epoch_values,
            writer=writer,
            learning_rate=learning_rate,
            step="Training",
        )

        with torch.no_grad():
            # Run evaluation.
            tr_params["net"].eval()
            epoch_values = run_one_epoch(
                loaders["val"],
                tr_params,
                writer,
                [current_epoch, tr_params["saved_epoch"]],
                no_of_epochs,
                device,
                norm_params['train'],
                classes_names,
                batchsize,
                desired_batchsize,
                log_path,
                model_name,
                step="Validation",
            )
            log_metrics(
                current_epoch,
                epoch_values,
                writer,
                learning_rate,
                step="Validation",
            )
            # 尝试：学习率固定衰减
            #  lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            # param_groups[0]表示第一个参数组，若没具体设定则为全局学习率，否则为第一个参数组的学习率，一般把第一个参数组设为分类头
            learning_rate_before = tr_params["optimizer"].param_groups[0]['lr']
            # 更新学习率调度器
            plateau_scheduler.step(epoch_values["loss"])

            # # 更新学习率
            # scheduler.step()
            learning_rate = tr_params["optimizer"].param_groups[0]['lr']
            logging.info(f"Learning rate now is {learning_rate}")

            # if learning_rate<=0.0001:
            #     controller.unfreez_all()


            # # 早停策略
            # early_stopping(epoch_values["loss"])
            # if early_stopping.stop_training:
            #     logging.info(f"Early stopping at epoch {current_epoch}")
            #     print(f"Early stopping at epoch {current_epoch}")
            #     break

            if model_path is not None:
                model_last_path = (log_path / model_path).with_name("model_last.pth")
                model_best_path = (log_path / model_path).with_name("model_best.pth")
            else:
                model_last_path = log_path / "texture_model/model_last.pth"
                model_best_path = log_path / "texture_model/model_best.pth"


            if not os.path.exists(model_last_path.parent):
                os.makedirs(model_last_path.parent)

            # 每一轮都保存当前模型，覆盖 model_last.pth
            model.save_model(
                current_epoch + 1,
                tr_params["net"].state_dict(),
                epoch_values["loss"],
                tr_params["optimizer"].state_dict(),
                tr_params["scaler"].state_dict(),
                model_last_path,
            )
            logging.info("Latest model (epoch %d) saved to %s", current_epoch + 1, model_last_path)

            # 判断是否为最佳模型，保存为 model_best.pth
            if epoch_values["loss"] < tr_params["best_loss"]:
                tr_params["best_loss"] = epoch_values["loss"]

                model.save_model(
                    current_epoch + 1,
                    tr_params["net"].state_dict(),
                    epoch_values["loss"],
                    tr_params["optimizer"].state_dict(),
                    tr_params["scaler"].state_dict(),
                    model_best_path,
                )
                logging.info("Best model (epoch %d) saved to %s", current_epoch + 1, model_best_path)



    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished training in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )

# 在每个验证周期后检查模型性能，并在一定的次数内性能没有提升时停止训练。这有助于防止过拟合并节省计算资源
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


class TrainingPhaseController:
    def __init__(self, model,strategy="phased"):
        self.model = model
        self.strategy = strategy
        self.phase = 0

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False
    def unfreez_all(self):
        for p in self.model.parameters():
            p.requires_grad = True
        print(f"Unfreezing all layers")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Frozen params: {total_params - trainable_params}")


    def unfreeze_encoders(self):
        for name, module in self.model.transformer.encoder.named_modules():
            for p in module.parameters():
                p.requires_grad = True
    #
    #
    # def unfreeze_embeddings(self):
    #     for name, module in self.model.transformer.embeddings.named_modules():
    #         for p in module.parameters():
    #             p.requires_grad = True


    def set_phase(self):
        self.phase+=1
        print(f"Switching to training phase {self.phase}")
        self.freeze_all()
        if self.phase == 1:
            self.unfreeze_encoders()
            print(f"Unfreezing encoders")

        elif self.phase == 2:
            self.unfreeze_embeddings()
            print(f"Unfreezing embeddings")
        for name, param in self.model.named_parameters():
            print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

