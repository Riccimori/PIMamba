#coding=utf8
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import random
from abc import ABC
from typing import Dict, Optional, Tuple, Callable, List, cast, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from tqdm import tqdm

try:
    from .utils import HiddenPrints
except:
    from utils import HiddenPrints


# class ImgFloader:
# class ImgFloader2:
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Single_Dataset(Dataset):

    def __init__(self, data_root, images_name, images_label, is_test=False):
        super(Single_Dataset, self).__init__()
        self.images_name = images_name
        self.images_label = images_label
        self.data_root =data_root

        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.cutout_prob = 1
        self.cutout = Cutout(n_holes=1,length=26)

        self.is_test = is_test

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, item):

        images_path = os.path.join(self.data_root,self.images_name[item])

        with open(images_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # img.convert('L')
        img = self.to_tensor(img)
        if not self.is_test and torch.rand(1) < self.cutout_prob:
            img = self.cutout(img)
        # img = self.norm(img)
        label_str = self.images_label[item]

        
        #WEITIAO
        label_list = [int(char) for char in label_str]

        # 将整数列表转换为张量
        label = torch.tensor(label_list, dtype=torch.float)

        #yu
        # label_bits = [int(char) for char in label_str[:11]]  # 只取前11位
        
        # label = label_bits.index(1)  # 100 -> 0, 010 -> 1, 001 -> 2

        if self.is_test:
            return img
        return img, label



def build_dataset(data_root,images_name,images_label,is_test=False):
    dataset = Single_Dataset(data_root = data_root, images_name = images_name, images_label = images_label, is_test = is_test)

    return dataset


if __name__ == '__main__':

    # data_root= r'C:\rsh\Paper\Oil_Diagram\Data\5.1\224\ALL_224_In'
    # df = pd.read_csv(r'C:/rsh/Paper/Oil_Diagram/Data/5.1/data_inf1.csv',encoding='gbk',dtype=str)

    data_root = r'C:\rsh\Paper\Oil_Diagram\Data\Priori_Data\Priori_ALL_In'
    df = pd.read_csv(r'C:\rsh\Paper\Oil_Diagram\Data\Priori_Data/Priori_data_inf.csv',encoding='gbk',dtype=str)
    selected_columns = ['Img_name','Label']
    img_data = df[selected_columns]

    first_row = pd.DataFrame(img_data.iloc[[0]])  # ��ȡ��һ������
    remaining_rows = img_data.iloc[1:].reset_index(drop=True) # ��ȡ����һ���������������

    train_data = pd.concat([first_row, remaining_rows.sample(frac=0.8,random_state=1)])
    val_data = pd.concat([first_row, remaining_rows.drop(train_data.index)])

    train_data_Image = train_data['Img_name'][0:].values
    train_label = train_data['Label'][0:].values
    val_data_Image = val_data['Img_name'][0:].values
    val_label = val_data['Label'][0:].values
    print(train_data_Image)
    b = Single_Dataset(data_root,train_data_Image,train_label)
    print(len(b))
    print(b[0][0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(b[0][0].permute(1, 2, 0))  # 转置维度以正确显示
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    plt.show()

    # for img_num in b:
    #     img_path = os.path.join(os.path.join(img_num., self.images_name[item]))
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     axes[0].imshow(img_without_cutout.permute(1, 2, 0))  # 转置维度以正确显示
    #     axes[0].set_title('Original Image')
    #     axes[0].axis('off')



    # for i in tqdm(b):
    #     q = i
    #     pass