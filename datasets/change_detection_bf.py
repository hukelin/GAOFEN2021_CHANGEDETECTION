from torch.utils.data import DataLoader
import datasets.transform as tr
# import transform as tr
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChangeDetection(Dataset):
    # 类别数
    # CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']
    CLASSES = ['未变化区域', '建筑物']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection, self).__init__()
        # self.root = os.path.join(root, 'ChangeDetection')
        self.root = root
        self.mode = mode
        self.use_pseudo_label = use_pseudo_label

        if mode in ['train', 'val', 'pseudo_labeling']:
            # 训练数据集文件夹
            self.root = os.path.join(self.root, 'train')
            # 读取训练图像
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            # 训练图像排序
            self.ids.sort()
            # 划分数据集-训练集:验证集 = 9:1
            if mode == 'val':
                self.ids = self.ids[::10]
            else:
                self.ids = list(set(self.ids) - set(self.ids[::10]))
        else:
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
        self.ids.sort()

        # 数据增强
        self.transform = transforms.Compose([
            tr.RandomFlipOrRotate()
        ])

        # 数据标准化
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = Image.open(os.path.join(self.root, 'im1', id))
        img2 = Image.open(os.path.join(self.root, 'im2', id))

        if self.mode == "test":
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
            return img1, img2, id

        if self.mode == "val":
            mask1 = Image.open(os.path.join(self.root, 'label1', id))
            mask2 = Image.open(os.path.join(self.root, 'label2', id))
        else:
            if self.mode == 'pseudo_labeling' or (self.mode == 'train' and not self.use_pseudo_label):
                mask1 = Image.open(os.path.join(self.root, 'label1', id))
                mask2 = Image.open(os.path.join(self.root, 'label2', id))
            else:
                mask1 = Image.open(os.path.join('outdir/masks/train/im1', id))
                mask2 = Image.open(os.path.join('outdir/masks/train/im2', id))

            if self.mode == 'train':
                gt_mask1 = np.array(Image.open(
                    os.path.join(self.root, 'label1', id)))
                mask_bin = np.zeros_like(gt_mask1)
                # 原始标签图中未变化区域的值为0
                # 二值分类中，未变化为1，变化为0
                mask_bin[gt_mask1 == 0] = 1
                mask_bin = Image.fromarray(mask_bin)

                sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                         'mask_bin': mask_bin})
                img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                    sample['mask2'], sample['mask_bin']

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        if self.mode == 'train':
            mask_bin = torch.from_numpy(np.array(mask_bin)).float()
            return img1, img2, mask1, mask2, mask_bin

        return img1, img2, mask1, mask2, id

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    trainset = ChangeDetection(
        root="/home/kelin/data", mode="train")
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True,
                             pin_memory=False, num_workers=4, drop_last=True)
    print(len(trainloader))
    for i in trainloader:
        print(i[3].max(), i[3].min())
        break
