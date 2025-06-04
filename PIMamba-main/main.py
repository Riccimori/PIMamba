import argparse
import datetime
import os.path

import numpy as np
import time

import timm
import torch
import torch.backends.cudnn as cudnn
from timm.scheduler import create_scheduler
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from pathlib import Path
import csv

from timm.optim import create_optimizer
from timm.utils import NativeScaler

from utils import *
# from models.vmamba_patch import *
# from models.PIMamba import *
from models.vmamba import *

def get_args_parser():
    parser = argparse.ArgumentParser('EfficientFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='efficientformer_l1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--nb-classes', default=13, type=int, help='classes number')

    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument(
    #     '--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float,
    #                     default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu',
    #                     action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--clip-grad', type=float, default=0.01, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 2e-3)')

    # Dataset parameters
    parser.add_argument('--data_type', default='dual', type=str,
                        help='dataset type')
    # parser.add_argument('--data_root', default='/root/autodl-tmp/Oil_Dram/Data/Priori_Data/Priori_ALL_In', type=str,
    #                     help='dataset path')
    parser.add_argument('--data_root', default=r'/root/autodl-tmp/Oil_Dram/Data/ModiEnhance/ALL_In', type=str,
                        help='dataset path')
    # parser.add_argument('--inf_csv', default=r'/root/autodl-tmp/Oil_Dram/Data/Priori_Data/Priori_data_inf.csv', type=str,
    #                     help='information_csv')
    parser.add_argument('--inf_csv', default=r'/root/autodl-tmp/Oil_Dram/Data/ModiEnhance/data_inf.csv', type=str,
                        help='information_csv')
    parser.add_argument('--output_dir', default='exps',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--shuffle', action='store_false',
                        help='Dataloader shuffle')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    #Dataloader的工作线程数
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--train_rate', type=float, default=0.02,
                        help='train data rate')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.set_defaults(sync_bn=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    fieldname = ["train_lr", "train_ExactMatchRatio","train_MAP","train_AUC","train_loss", "epoch", "n_parameters", "test_loss", "test_ExactMatchRatio","test_MAP","test_AUC"]

    # �Ƿ���ϴζϵ㴦ִ��
    first = False if args.resume else True

    device = torch.device(args.device)

    # fix the seed for reproducibilityijij
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    kwargs = dict(
        img_size=args.input_size
    )

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = AsymmetricLoss()
    criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    # #不需要先验模型
    # model = PIMamba_tiny_s1l8()
    # model.to(device)

    # model = vmamba_small_s1l20()
    # model.to(device)


 # ------------------------------------需要先验模型
    model = PIMamba_tiny_s2l5()
    

    checkpoint_path = (r'/root/autodl-tmp/Oil_Dram/Code/VMamba-main/classification/table6/kuo_PI_tiny_s2l5_Pri_13w.pth')
    try:
        checkpoint = torch.load(checkpoint_path)
        # print(f"Loaded state_dict keys: {list(checkpoint['model'].keys())}")

        # 加载模型权重，但忽略分类头
        model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and 'classifier.head.weight' not in k and 'classifier.head.bias' not in k}
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if 'classifier' not in k}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    
        # model.load_state_dict(model_dict, strict=False)



    except FileNotFoundError:
        print("The weights file was not found.")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
# # ---------------------------------------------------------------------------------------------------------

    if args.sync_bn and args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    optimizer = create_optimizer(args, model_without_ddp)
    model.to(device)

   
# ---------------------------------------------------------------------------------------------------------
    
    # 使用strict=False来忽略不匹配的键
    # model.load_state_dict(checkpoint, strict=False)
    #
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    #
    # if missing_keys:
    #     print(f"Missing keys: {missing_keys}")
    # if unexpected_keys:
    #     print(f"Unexpected keys: {unexpected_keys}")


    train_images_name, train_images_label, val_images_name, val_images_label = read_split_data(args.inf_csv, args.train_rate)
   
    print(args.data_root)
    dataset_train = Single_Dataset(data_root=args.data_root, images_name=train_images_name,
                                   images_label=train_images_label)
    dataset_val = Single_Dataset(data_root=args.data_root, images_name=val_images_name, images_label=val_images_label)

    data_loader_train = DataLoader(
        dataset_train,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['overall_acc']:.1f}%")
        return

    print(f"Start training from {args.start_epoch} for {args.epochs} epochs")
    start_time = time.time()
    max_over_accuracy = 0.0
    max_overall_map = 0.0
    max_overall_auc = 0.0
    max_F1score = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode,
            set_training_mode=True,  # keep in eval mode during finetuning
        )

        # lr_scheduler.step(epoch)

        if epoch % 4 == 3:
            test_stats = evaluate(data_loader_val, model, device,criterion)
            print(test_stats)
            print(
                f"Overall_Accuracy of the network on the {len(dataset_val)} test images: {test_stats['ExactMatchRatio']:.1f}%")
            max_overall_acc = max(max_over_accuracy, test_stats["ExactMatchRatio"])
            max_overall_map = max(max_overall_map, test_stats["MAP"])
            max_overall_auc = max(max_overall_auc, test_stats["AUC"])
            print(f'Max accuracy: {max_overall_acc:.2f}%')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         **{f'test_{k}': v for k, v in test_stats.items()}}


            if args.output_dir and max_overall_acc == test_stats["ExactMatchRatio"]:
                checkpoint_paths = [output_dir / 'fine2_PI_tiny_s2l5_13w.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'ExactMatchRatio' : max_over_accuracy,
                    }, checkpoint_path)
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if first:
                if os.path.exists(output_dir / "fine2_PI_tiny_s2l5_13w.csv"):
                    os.remove(output_dir / "fine2_PI_tiny_s2l5_13w.csv")

            with open(output_dir / "fine2_PI_tiny_s2l5_13w.csv", "a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldname)
                if first:
                    writer.writeheader()
                    first = False

                writer.writerow(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
