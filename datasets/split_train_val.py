# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:41:11 2021

@author: DELL
"""

import glob
from shutil import copyfile
import os

image_paths1 = glob.glob("/home/kelin/data/train/im1/*.png")
image_paths2 = glob.glob("/home/kelin/data/train/im2/*.png")
# label_paths1 = glob.glob("/home/kelin/data/train/label1/*.png")
# label_paths2 = glob.glob("/home/kelin/data/train/label2/*.png")
label_paths = glob.glob("/home/kelin/data/train/label/*.png")

# k fold交叉验证
k = 5
for fold in range(k):
    fold_train_image_dir1 = "./data/train_"+str(fold)+"/im1/"
    fold_train_image_dir2 = "./data/train_"+str(fold)+"/im2/"
    fold_train_label_dir1 = "./data/train_"+str(fold)+"/label1/"
    fold_train_label_dir2 = "./data/train_"+str(fold)+"/label2/"
    fold_train_label_dir = "./data/train_"+str(fold)+"/label/"

    fold_val_image_dir1 = "./data/val_"+str(fold)+"/im1/"
    fold_val_image_dir2 = "./data/val_"+str(fold)+"/im2/"
    fold_val_label_dir1 = "./data/val_"+str(fold)+"/label1/"
    fold_val_label_dir2 = "./data/val_"+str(fold)+"/label2/"
    fold_val_label_dir = "./data/val_"+str(fold)+"/label/"

    if not os.path.exists(fold_train_image_dir1):
        os.makedirs(fold_train_image_dir1)
    if not os.path.exists(fold_train_image_dir2):
        os.makedirs(fold_train_image_dir2)
    if not os.path.exists(fold_train_label_dir1):
        os.makedirs(fold_train_label_dir1)
    if not os.path.exists(fold_train_label_dir2):
        os.makedirs(fold_train_label_dir2)
    if not os.path.exists(fold_train_label_dir):
        os.makedirs(fold_train_label_dir)

    if not os.path.exists(fold_val_image_dir1):
        os.makedirs(fold_val_image_dir1)
    if not os.path.exists(fold_val_image_dir2):
        os.makedirs(fold_val_image_dir2)
    if not os.path.exists(fold_val_label_dir1):
        os.makedirs(fold_val_label_dir1)
    if not os.path.exists(fold_val_label_dir2):
        os.makedirs(fold_val_label_dir2)
    if not os.path.exists(fold_val_label_dir):
        os.makedirs(fold_val_label_dir)

    for i in range(len(image_paths1)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == fold:
            # 获取时相1的图像
            image_path1 = image_paths1[i]
            # 获取时相2的图像
            image_path2 = image_path1.replace("im1", "im2")

            # 获取图像名称
            fold_val_image_path1 = fold_val_image_dir1 + \
                os.path.basename(image_path1)

            fold_val_image_path2 = fold_val_image_dir2 + \
                os.path.basename(image_path2)

            copyfile(image_path1, fold_val_image_path1)
            copyfile(image_path2, fold_val_image_path2)

            label_path = label_paths[i]

            fold_val_label_path = fold_val_label_dir + \
                os.path.basename(image_path2)
            copyfile(label_path, fold_val_label_path)

            label_path1 = label_path.replace("label", "label1")

            fold_val_label_path1 = fold_val_label_dir1 + \
                os.path.basename(image_path2)
            copyfile(label_path1, fold_val_label_path1)

            label_path2 = label_path.replace("label", "label2")
            fold_val_label_path2 = fold_val_label_dir2 + \
                os.path.basename(image_path2)
            copyfile(label_path2, fold_val_label_path2)
        else:
            image_path1 = image_paths1[i]
            image_path2 = image_path1.replace("im1", "im2")

            # 获取图像名称
            fold_train_image_path1 = fold_train_image_dir1 + \
                os.path.basename(image_path1)
            fold_train_image_path2 = fold_train_image_dir2 + \
                os.path.basename(image_path2)

            copyfile(image_path1, fold_train_image_path1)
            copyfile(image_path2, fold_train_image_path2)
            label_path = label_paths[i]

            fold_train_label_path = fold_train_label_dir + \
                os.path.basename(image_path2)
            copyfile(label_path, fold_train_label_path)

            label_path1 = label_path.replace("label", "label1")

            fold_train_label_path1 = fold_train_label_dir1 + \
                os.path.basename(image_path2)
            copyfile(label_path1, fold_train_label_path1)

            label_path2 = label_path.replace("label", "label2")
            fold_train_label_path2 = fold_train_label_dir2 + \
                os.path.basename(image_path2)
            copyfile(label_path2, fold_train_label_path2)
