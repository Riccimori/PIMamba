# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in ori_main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
from timm.utils import ModelEma,accuracy

from .utils import MetricLogger, SmoothedValue
from .metric import *


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        if isinstance(samples, list):
            samples = [i.to(device, non_blocking=True) for i in samples]
        else:
            samples = samples.to(device, non_blocking=True)
        if isinstance(targets, list):
            targets = [i.to(device, non_blocking=True) for i in targets]
        else:
            targets = targets.to(device, non_blocking=True)

        if True:  # with torch.cuda.amp.autocast():
            outputs = model(samples)
            # print("outputs.shape",outputs.shape)
            # print("targets.shape",targets.shape)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        # acc1 = m_accuracy(output, target)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        sigmoid = nn.Sigmoid()
        metric_outputs = sigmoid(outputs)


        # DUO
        threshold = 0.7
        # overall_acc = mul_calculate_overall_accuracy(targets,outputs,threshold)
        ExactMatchRatio = mul_calculate_ExactMatchRatio(targets, metric_outputs, threshold)
        map = sk_calculate_map(targets, metric_outputs)
        auc = sk_calculate_auc(targets, metric_outputs)

        # DAN
        # ExactMatchRatio = sig_calculate_overall_accuracy(targets, outputs)
        # # map = sk_calculate_map(targets, outputs)
        # auc = sk_calculate_auc(targets, outputs)

        # if not isinstance(precision, torch.Tensor):
        #     # 如果 precision 是整数，将其转换为张量
        #     precision = torch.tensor(precision)

        metric_logger.meters['ExactMatchRatio'].update(ExactMatchRatio.item(), n=targets.shape[0])
        metric_logger.meters['MAP'].update(map.item(), n=targets.shape[0])
        metric_logger.meters['AUC'].update(auc.item(), n=targets.shape[0])
        # metric_logger.meters['F1_score'].update(F1_score.item(), n=targets.shape[0])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,criterion):

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        if isinstance(images, list):
            images = [i.to(device, non_blocking=True) for i in images]
        else:
            images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(device, non_blocking=True) for i in target]
        else:
            target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        # acc1, _ = accuracy(output, target, topk=(1, 5))
        sigmoid = nn.Sigmoid()
        metric_output = sigmoid(output)

        threshold = 0.7
        # # overall_acc = mul_calculate_overall_accuracy(targets,outputs,threshold)
        ExactMatchRatio = mul_calculate_ExactMatchRatio(target, metric_output, threshold)
        map = sk_calculate_map(target, metric_output)
        auc = sk_calculate_auc(target, metric_output)

        # ExactMatchRatio = sig_calculate_overall_accuracy(target, output)
        # map = sk_calculate_map(target, output)
        # auc = sk_calculate_auc(target, output)



        batch_size = target.shape[0]
        metric_logger.update(loss=loss.item())

        metric_logger.meters['ExactMatchRatio'].update(ExactMatchRatio.item(),  n=batch_size)
        metric_logger.meters['MAP'].update(map.item(), n=batch_size)
        metric_logger.meters['AUC'].update(auc.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* ExactMatchRatio@ {ExactMatchRatio.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(ExactMatchRatio=metric_logger.ExactMatchRatio, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    res = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return res

