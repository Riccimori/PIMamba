# load libraries
import time
import datetime

#import miseval.confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#from miseval import evaluate, calc_ConfusionMatrix, calc_IoU


def mul_calculate_ExactMatchRatio(y_true, y_pred,threshold):

    # Calculate accuracy

    # correct_predictions = (y_pred == y_true).all(dim=1).float()  # Check if all elements are correct in each sample
    predictions_bool = (y_pred >= threshold).float()
    correct_predictions = (predictions_bool == y_true).all(dim=1).float()
    accuracy = correct_predictions.sum() / y_true.size(0)
    # print(correct_predictions.sum())
    # print(y_true.size(0))
    return accuracy

def sig_calculate_overall_accuracy(y_true, y_pred):
    # _,label_indices = y_true.max(1)
    # _, predicted = y_pred.max(1)
    # return (predicted == label_indices).float().mean()
    predicted = y_pred.argmax(1)  # (batch,)
    return (predicted == y_true).float().mean()



def calculate_accuracy(y_true, y_pred,threshold):
    """
    Calculate the Jaccard Similarity (IoU) for multi-label classification.

    Parameters:
    - y_true (torch.Tensor): The ground truth labels tensor, where each element is 0 or 1.
                             Shape: [batch_size, num_labels]
    - y_pred (torch.Tensor): The prediction labels tensor, where each element is 0 or 1.
                             Shape: [batch_size, num_labels]

    Returns:
    - accuracy (torch.Tensor): The computed mean Jaccard Similarity (IoU).
    """
    y_true_bool = (y_true> threshold).int()
    y_pred_bool = (y_pred > threshold).int()
    intersection = (y_true_bool & y_pred_bool).float().sum(dim=1)  # Compute intersection
    union = (y_true_bool | y_pred_bool).float().sum(dim=1)  # Compute union

    # Avoid division by zero by adding a small epsilon where union is zero
    epsilon = 1e-7
    iou = intersection / (union + epsilon)  # Compute IoU

    # Calculate mean IoU across all samples
    accuracy = iou.mean()

    return accuracy

def sig_calculate_precision(y_true, y_pred,threshold):
    precision_score(y_true,y_pred)


def calculate_precision(y_true, y_pred,threshold):
    """
    Calculate the average precision score across all labels.
    """
    device = y_pred.device
    y_true = y_true.to(device)
    # pred_sum = pred.sum(dim=1).float()  # Sum of predictions for each sample
    # correct_pred = (pred & target).sum(dim=1).float()  # Sum of correct predictions for each sample
    #
    # # Handle cases where the denominator would be zero (i.e., no positive predictions)
    # valid = pred_sum > 0  # Boolean tensor indicating samples with at least one positive prediction
    # ratios = torch.zeros_like(pred_sum)  # Initialize a tensor for storing precision ratios
    # ratios[valid] = correct_pred[valid] / pred_sum[valid]  # Compute precision only for valid cases
    #
    # # Calculate the average precision across all samples
    # # Avoid dividing by zero by ensuring at least one valid case exists
    # precision = ratios.sum() / torch.max(torch.tensor(1.0, device=device), valid.sum().float())
    y_pred_bool = (y_pred > threshold).int()
    y_true_bool = (y_true > threshold).int()
    correct_pred = (y_true_bool & y_pred_bool).float().sum(dim=1)

    # Calculate the number of positive predictions per sample
    pred_positives = y_pred_bool.sum(dim=1).float()

    # Avoid division by zero by using a mask for samples with non-zero predictions
    valid = pred_positives > 0
    precision_scores = torch.zeros_like(correct_pred)
    precision_scores[valid] = correct_pred[valid] / pred_positives[valid]

    # Calculate the average precision across all samples
    precision = precision_scores.sum() / valid.float().sum()

    return precision

def mul_calculate_recall(y_true, y_pred, threshold=0.9):
    # Apply the threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).float()

    # Calculate True Positives (TP) and False Negatives (FN) for each class
    TP = (y_true * y_pred_binary).sum(dim=1)
    FN = (y_true * (1 - y_pred_binary)).sum(dim=1)

    # Calculate recall for each class
    recall_per_class = TP / (TP + FN + 1e-10)  # Adding a small epsilon to avoid division by zero

    # Calculate average recall across all classes
    recall_average = recall_per_class.mean()

    return recall_per_class, recall_average

def calculate_recall(y_true, y_pred,threshold):
    """
    Calculate the average recall score across all labels.
    """
    y_true_bool = (y_true > threshold).int()
    y_pred_bool = (y_pred > threshold).int()
    correct_pred = (y_true_bool & y_pred_bool).float().sum(dim=1)

    # Calculate the number of actual positive labels per sample
    true_positives = y_true_bool.sum(dim=1).float()

    # Avoid division by zero by using a mask for samples with non-zero true positives
    valid = true_positives > 0
    recall_scores = torch.zeros_like(correct_pred)
    recall_scores[valid] = correct_pred[valid] / true_positives[valid]

    # Calculate the average recall across all samples
    recall = recall_scores.sum() / valid.float().sum()

    return recall

def matrix(y_true, y_pred,threshold):
    y_true_bool = (y_true > threshold).int()
    y_pred_bool = (y_pred > threshold).int()

def sig_calculate_f1_score(y_true, y_pred,threshold):
    return f1_score(y_true,y_pred)


def calculate_f1_score(y_true, y_pred,threshold):
    """
    Calculate the average F1 score across all labels.
    """
    y_true_bool = (y_true > threshold).int()
    y_pred_bool = (y_pred > threshold).int()
    correct_pred = (y_true_bool & y_pred_bool).float().sum(dim=1)

    # Calculate the sum of true and predicted positives per sample
    total_positives = y_true_bool.sum(dim=1).float() + y_pred_bool.sum(dim=1).float()

    # Avoid division by zero by using a mask for samples with non-zero total positives
    valid = total_positives > 0
    f1_scores = torch.zeros_like(correct_pred)
    f1_scores[valid] = 2 * correct_pred[valid] / total_positives[valid]

    # Calculate the average F1 score across all samples
    f1_score = f1_scores.sum() / valid.float().sum()

    return f1_score
# 平均类型: 这里我们使用了 'macro' 平均来计算精确度、召回率和F1分数，这意味着会为每个标签分别计算指标，然后取所有标签指标的平均值。这种方法不考虑标签的不平衡性。如果需要，可以使用 'micro' 或 'weighted' 来代替，这两者在处理不平衡类别时更为合适。
# ROC-AUC: 计算 ROC-AUC 时，需要预测值是概率而不是二进制标签。此外，如果某个标签的真实值或预测值中只包含同一类，roc_auc_score 函数将会报错，因此在计算之前需要进行检查。

from sklearn.metrics import roc_auc_score, average_precision_score


def sk_calculate_map(y_true, y_pred):

    y_prob = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    average_precisions = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # 如果标签有正样本
            average_precisions.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            average_precisions.append(0)

    # 计算mAP
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP


def sk_calculate_auc(y_true, y_pred):
    y_prob = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    # 计算每个标签的AUC
    aucs = []

    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # 如果标签有正样本
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        else:
            aucs.append(0)

    # 计算macro AUC
    macro_auc = sum(aucs) / len(aucs)

    return macro_auc


def calculate_auc(y_true, y_pred_prob):
    y_true = y_true.float()
    y_pred_prob = y_pred_prob.float()
    n_classes = y_true.shape[1]

    aucs = []
    for i in range(n_classes):
        true = y_true[:, i]
        pred = y_pred_prob[:, i]

        if torch.sum(true) == 0 or torch.sum(true) == true.size(0):
            # 如果所有真实值都为0或1，AUC没有意义，跳过这个标签
            aucs.append(0.5)  # 在sklearn中，这种情况的AUC通常定义为0.5
            continue

        desc_score_indices = torch.argsort(pred, descending=True)
        true = true[desc_score_indices]

        # 计算累积的真正例和假正例
        cum_true_positive = torch.cumsum(true, dim=0)
        false_positive_rate = torch.cumsum(1 - true, dim=0) / (true.size(0) - cum_true_positive[-1])
        true_positive_rate = cum_true_positive / cum_true_positive[-1]

        # 在开头插入(0,0)点
        true_positive_rate = torch.cat([torch.tensor([0]), true_positive_rate])
        false_positive_rate = torch.cat([torch.tensor([0]), false_positive_rate])

        # 使用梯形法则计算AUC
        auc = torch.trapz(true_positive_rate, false_positive_rate)
        aucs.append(auc.item())

    return torch.tensor(aucs).mean().item()


# 定义计算AP的函数
def calculate_map(y_true, y_pred_prob):
    y_true = y_true.float()
    y_pred_prob = y_pred_prob.float()
    n_classes = y_true.shape[1]

    aps = []
    for i in range(n_classes):
        true = y_true[:, i]
        pred = y_pred_prob[:, i]

        if torch.sum(true) == 0:
            aps.append(0.0)
            continue

        desc_score_indices = torch.argsort(pred, descending=True)
        true = true[desc_score_indices]

        # 计算累积的真正例
        cum_true_positive = torch.cumsum(true, dim=0)
        precision = cum_true_positive / (torch.arange(1, true.size(0) + 1).float())

        # 计算平均精度
        ap = torch.sum(precision * true) / torch.sum(true)
        aps.append(ap.item())

    return torch.tensor(aps).mean().item()


if __name__ == '__main__':
    y_pred = torch.tensor([[ -8.1625,  -5.9761,  -5.8266,  -6.5001, -11.9222,  -5.5530, -13.8096,
           0.9839,  -5.8033,  -7.7272, -10.7767],
        [  6.7053,  -2.7667,  -4.4205,  -4.0083, -11.0315,  -4.5189,   3.3456,
         -15.3946,  -0.4328, -10.1185, -10.8739],
        [ -6.1746,   4.2850,   5.9153,  -8.9775,  -7.4671,  -6.4137,  -5.0634,
          -4.5515,  -4.4240, -13.8995,  -7.8071],
        [-11.5520, -10.1957,  -6.3126,  -7.7491,  -6.5679,  -6.5868, -19.8589,
           7.8001,  -9.8962,  -8.9129, -11.0903],
        [-43.4542, -21.7422, -16.2883, -33.4383, -24.4977, -27.8221, -23.9754,
         -19.3235, -33.5549, -21.1094,  36.9360],
        [  0.2699,  -7.4380, -12.1081,  14.5977, -12.5414, -12.3875, -16.4702,
          -8.9434,   5.1927, -12.4190, -22.1772],
        [-37.0059, -24.9798, -20.4582, -26.1801, -27.4706, -28.0043, -13.4983,
         -20.6950, -27.5167, -19.7477,  32.4741],
        [-17.6197, -20.1466, -22.2231,  -5.5701, -14.8688, -10.6307,  -8.2372,
         -27.6458, -17.5691,   9.3510, -13.1634]]
)
    y_true = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]
)
    # y_pred = torch.tensor([[0.1, -0.1, 2.8], [2.1, 7.8, 0.1],[3.7, 3.2, 3.1]])
    # y_ture = torch.tensor([[1, 0, 0], [0, 1, 0],[1, 0, 0]])

    sigmoid = nn.Sigmoid()
    prediction = sigmoid(y_pred)
    # precision = mul_calculate_overall_accuracy(y_true, prediction,0.9)

    sk_auc = sk_calculate_auc(y_true, prediction)
    sk_map = sk_calculate_map(y_true, prediction)
    auc = calculate_auc(y_true, prediction)
    map = calculate_map(y_true, prediction)
    # precision = mul_calculate_overall_accuracy(y_true,y_pred,1)
    print("-----")