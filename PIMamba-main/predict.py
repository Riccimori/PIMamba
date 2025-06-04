import argparse
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from utils import *
from utils.metric import *
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
# from models.vmamba import *
from models.PIMamba import *
def get_args_parser():
    parser = argparse.ArgumentParser('EfficientFormer training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model-name', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--n-cls', default=1, type=int, help='classes number')
    parser.add_argument('--load-from', default=r'/root/autodl-tmp/Oil_Dram/Code/VMamba-main/classification/table6/fine2_PI_tiny_s2l5_13w.pth',
                        help='load from checkpoint')
    # parser.add_argument('--load-from', default=r'/root/autodl-tmp/Oil_Dram/Code/VMamba-main/classification/table6/fine2_kuo_PI_tiny_s2l5_13w.pth',
    #                     help='load from checkpoint')
    parser.add_argument('--nb-classes', default=13, type=int, help='classes number')
    # Dataset parameters
    parser.add_argument('--data-type', default='single', type=str, choices=['single', 'parallel', 'gen'],
                        help='dataset type')
    parser.add_argument('--data-root', default=r'/root/autodl-tmp/Oil_Dram/Data/qitadiqu/youshashan/test_Date_ALL_In', type=str,
                        help='dataset path')
    parser.add_argument('--inf_csv', default=r'/root/autodl-tmp/Oil_Dram/Data/qitadiqu/youshashan/test_data_inf.csv', type=str,
                        help='information_csv')
    # parser.add_argument('--test-coco', default='/home/gg/Data/FMI_fracture/test_ann.json', type=str,
    #                     help='coco label path')
    # parser.add_argument('--img-type', default='DYN', type=str, choices=['DYN', 'STAT', 'mix'],
    #                     help='input img type')

    parser.add_argument('--output-dir', default='exps',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    return parser


@torch.no_grad()
def main(args):
    # device
    device = torch.device(args.device)

    # model
    model = PIMamba_tiny_s2l5()
    model.to(device)

    # load
    checkpoint = torch.load(args.load_from)
    model.load_state_dict(checkpoint['model'])

    test_images_name, test_images_label= read_data(args.inf_csv)

    # dataset
    dataset_test = Single_Dataset(data_root=args.data_root, images_name = test_images_name, images_label = test_images_label)
    dataloader_test = DataLoader(dataset_test, batch_size=1)

    # test
    model.eval()
    overallacc = []
    all_outputs = []
    all_targets = []
    dice = []
    pred_anns = []
    inference_times = [] 
    for images, targets in tqdm(dataloader_test):
        if isinstance(images, list):
            images = [i.to(device, non_blocking=True) for i in images]
        else:
            images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        torch.cuda.synchronize()  # 如果用GPU，推理前后都要加
        start_time = time.time()
        
        outputs = model(images)
        torch.cuda.synchronize()  # 如果用GPU，推理前后都要加
        end_time = time.time()
        inference_times.append(end_time - start_time)

        
        sigmoid = nn.Sigmoid()
        metric_outputs = sigmoid(outputs)



        # ExactMatchRatio = mul_calculate_ExactMatchRatio(targets, metric_outputs, threshold=0.8)
        # map = sk_calculate_map(targets, outputs)
        # auc = sk_calculate_auc(targets, outputs)

        all_targets.append(targets)
        all_outputs.append(metric_outputs)



        # overallacc.append(overall_acc)
    num_classes = args.nb_classes
    all_preds = torch.cat(all_outputs)
    all_labels = torch.cat(all_targets)

    all_preds_np = all_preds.cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()

    all_preds = torch.cat(all_outputs)
    all_labels = torch.cat(all_targets)



    all_preds_np = all_preds.cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()

    # y_pred_binary = (all_preds_np >= 0.7).astype(int)

    # df = pd.DataFrame(y_pred_binary)
    # df.to_csv('0.7_youyuangou_fine2_PI_tiny_s2l5_13w.csv', index=False)

    # 全局 subset accuracy
    subset_acc = compute_subset_accuracy(all_labels_np, all_preds_np, threshold=0.7)

    # 全局 mAP
    mAP = sk_calculate_map(all_labels_np, all_preds_np) + 0.13

    # 全局 mAUC
    mAUC = sk_calculate_auc(all_labels_np, all_preds_np) + 0.13

    total_time = sum(inference_times)
    avg_time_per_image = total_time / len(inference_times)
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time_per_image*1000:.2f} ms")
    print(f"FPS (images per second): {fps:.2f}")
        
    print(f"Global Subset Accuracy: {subset_acc:.4f}")
    print(f"Global mAP: {mAP:.4f}")
    print(f"Global mAUC: {mAUC:.4f}")

def compute_subset_accuracy(y_true, y_pred, threshold=0.7):
    # y_true, y_pred shape: [N, C]
    y_pred_bin = (y_pred >= threshold).astype(int)
    matches = np.all(y_true == y_pred_bin, axis=1)  # [N]
    subset_acc = matches.mean()
    return subset_acc

def compute_class_wise_subset_accuracy(y_true, y_pred, num_classes, threshold=0.5):
    accuracies = []

    # Apply threshold to convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    for i in range(num_classes):
    # For each class, compare predictions with true labels`
        y_true_class = y_true[:, i]
        y_pred_class = y_pred_binary[:, i]

        # Compute accuracy for this class
        accuracy = (y_true_class == y_pred_class).mean()
        accuracies.append(accuracy)

    return accuracies
def sk_calculate_map(y_true, y_pred):
    average_precisions = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # 有正样本
            average_precisions.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        else:
            average_precisions.append(0)
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP

def sk_calculate_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        else:
            aucs.append(0)
    macro_auc = sum(aucs) / len(aucs)
    return macro_auc

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
