# --------------------------------------------------------
# Modified By $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma
import io
import random
import time
from collections import defaultdict, deque
import datetime
import pandas as pd
import sys
import torch.distributed as dist


def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'max_accuracy_ema' in checkpoint:
            max_accuracy_ema = checkpoint['max_accuracy_ema']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_ema


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        #记录迭代时间和数据读取时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            # 'time: {time}',
            # 'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


# def replace_layernorm(net):
#     import apex
#     for child_name, child in net.named_children():
#         if isinstance(child, torch.nn.LayerNorm):
#             setattr(net, child_name, apex.normalization.FusedLayerNorm(
#                 child.weight.size(0)))
#         else:
#             replace_layernorm(child)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def read_split_data(inf_csv,train_rate):
    random.seed(0)
    assert os.path.exists(inf_csv), "information_csv: {} does not exist".format(inf_csv)
    df = pd.read_csv(inf_csv,encoding='gbk',dtype=str)
    selected_columns = ['Img_name','Label']
    img_data = df[selected_columns]

    first_row = pd.DataFrame(img_data.iloc[[0]])  # ��ȡ��һ������
    remaining_rows = img_data.iloc[1:].reset_index(drop=True) # ��ȡ����һ���������������

    # train_data = pd.concat([first_row, remaining_rows.sample(frac=0.9,random_state=1)])
    train_data = pd.concat([first_row, remaining_rows.sample(frac=0.8)])
    val_data = pd.concat([first_row, remaining_rows.drop(train_data.index)])

    train_data_Image = train_data['Img_name'][0:].values
    train_label = train_data['Label'][0:].values.tolist()
    val_data_Image = val_data['Img_name'][0:].values
    val_label = val_data['Label'][0:].values.tolist()

    return train_data_Image, train_label, val_data_Image, val_label


def read_test_data(inf_csv):
    random.seed(0)
    assert os.path.exists(inf_csv), "information_csv: {} does not exist".format(inf_csv)
    df = pd.read_csv(inf_csv,encoding='gbk',dtype=str)
    selected_columns = ['Img_name','Label']
    img_data = df[selected_columns]

    first_row = pd.DataFrame(img_data.iloc[[0]])
    remaining_rows = img_data.iloc[1:].reset_index(drop=True)

    test_data = pd.concat([first_row, remaining_rows.sample(frac=0.5,random_state=1)])


    test_data_Image = test_data['Img_name'][0:].values
    test_label = test_data['Label'][0:].values.tolist()

    return test_data_Image, test_label

def read_data(inf_csv):
    random.seed(0)
    assert os.path.exists(inf_csv), "information_csv: {} does not exist".format(inf_csv)
    df = pd.read_csv(inf_csv,encoding='gbk',dtype=str)
    selected_columns = ['Img_name','Label']
    img_data = df[selected_columns]

    test_data_Image = img_data['Img_name'].values
    print(test_data_Image)
    test_label = img_data['Label'].values.tolist()

    return test_data_Image, test_label