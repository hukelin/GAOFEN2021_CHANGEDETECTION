from sklearn.model_selection import KFold
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
from sklearn.model_selection import train_test_split


class ChangeDetection(Dataset):
    # 类别数
    # CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']
    CLASSES = ['未变化区域', "非建筑物", '建筑物']

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
            # 划分数据集-训练集:验证集 = 4:1
            self.train_ids, self.val_ids = train_test_split(
                self.ids, test_size=0.2, random_state=0)
            if mode == 'train':
                self.ids = self.train_ids
            else:
                self.ids = self.val_ids
        else:
            self.root = os.path.join(self.root)
            self.ids = os.listdir(self.root)
        self.ids.sort()

        # 数据增强：随机的翻转和旋转
        self.transform = transforms.Compose([
            tr.RandomFlipOrRotate()
        ])

        # 数据标准化
        self.normalize = transforms.Compose([
            # transforms.ColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor()
            # transforms.Normalize(
            #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        id = self.ids[index]
        if self.mode != "test":
            img1 = Image.open(os.path.join(self.root, 'im1', id))
            img2 = Image.open(os.path.join(self.root, 'im2', id))

        if self.mode == "test":
            img1 = Image.open(os.path.join(self.root, id))
            img2 = Image.open(os.path.join(self.root, id))
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
            return img1, img2, id

        if self.mode == "val":
            mask1 = Image.open(os.path.join(self.root, 'label1', id))
            mask2 = Image.open(os.path.join(self.root, 'label2', id))
            mask_bin = Image.open(os.path.join(self.root, 'change', id))
            # mask_bin = torch.from_numpy(np.array(mask_bin)).float()
            # mask_bin = mask_bin / 255.0
        else:
            if self.mode == 'pseudo_labeling' or (self.mode == 'train' and not self.use_pseudo_label):
                mask1 = Image.open(os.path.join(self.root, 'label1', id))
                mask2 = Image.open(os.path.join(self.root, 'label2', id))
            else:
                mask1 = Image.open(os.path.join('outdir/masks/train/im1', id))
                mask2 = Image.open(os.path.join('outdir/masks/train/im2', id))

            if self.mode == 'train':
                mask_bin = Image.open(os.path.join(self.root, 'change', id))
                # 数据增强
                sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                         'mask_bin': mask_bin})
                img1, img2, mask1, mask2, mask_bin = sample['img1'], sample[
                    'img2'], sample['mask1'], sample['mask2'], sample['mask_bin']
        # print(np.array(img2).shape) (512, 512, 3)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        mask1 = torch.from_numpy(np.array(mask1)).float()
        mask2 = torch.from_numpy(np.array(mask2)).float()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()
        mask1 = mask1 / 255.0
        mask2 = mask2 / 255.0
        mask_bin = mask_bin / 255.0
        # if self.mode == 'train':
        #     return img1, img2, mask1, mask2, mask_bin, id
        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)


# ******************************添加边界信息**********************************
# class ChangeDetection(Dataset):
#     # 类别数
#     # CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']
#     CLASSES = ['未变化区域', "非建筑物", '建筑物']

#     def __init__(self, root, mode, use_pseudo_label=False):
#         super(ChangeDetection, self).__init__()
#         # self.root = os.path.join(root, 'ChangeDetection')
#         self.root = root
#         self.mode = mode
#         self.use_pseudo_label = use_pseudo_label

#         if mode in ['train', 'val', 'pseudo_labeling']:
#             # 训练数据集文件夹
#             self.root = os.path.join(self.root, 'train')
#             # 读取训练图像
#             self.ids = os.listdir(os.path.join(self.root, "im1"))
#             # 训练图像排序
#             self.ids.sort()
#             # 划分数据集-训练集:验证集 = 4:1
#             self.train_ids, self.val_ids = train_test_split(
#                 self.ids, test_size=0.2, random_state=0)
#             if mode == 'train':
#                 self.ids = self.train_ids
#             if mode == 'val':
#                 self.ids = self.val_ids
#         else:
#             self.root = os.path.join(self.root)
#             self.ids = os.listdir(self.root)
#         self.ids.sort()

#         # 数据增强：随机的翻转和旋转
#         self.transform = transforms.Compose([
#             tr.RandomFlipOrRotate()
#         ])

#         # 数据标准化
#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
#                 transforms.ColorJitter(
#                     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
#         ])

#     def __getitem__(self, index):
#         id = self.ids[index]
#         if self.mode != "test":
#             img1 = Image.open(os.path.join(self.root, 'im1', id))
#             img2 = Image.open(os.path.join(self.root, 'im2', id))

#         if self.mode == "test":
#             img1 = Image.open(os.path.join(self.root, id))
#             img2 = Image.open(os.path.join(self.root, id))
#             img1 = self.normalize(img1)
#             img2 = self.normalize(img2)
#             return img1, img2, id

#         if self.mode == "val":
#             mask1 = Image.open(os.path.join(self.root, 'label1', id))
#             mask2 = Image.open(os.path.join(self.root, 'label2', id))
#             edge1 = Image.open(os.path.join(self.root, 'edge1', id))
#             edge2 = Image.open(os.path.join(self.root, 'edge2', id))
#             mask_bin = Image.open(os.path.join(self.root, 'change', id))
#             edge_bin = Image.open(os.path.join(self.root, 'change_edge', id))
#             # mask_bin = torch.from_numpy(np.array(mask_bin)).float()
#             # mask_bin = mask_bin / 255.0
#         else:
#             if self.mode == 'pseudo_labeling' or (self.mode == 'train' and not self.use_pseudo_label):
#                 mask1 = Image.open(os.path.join(self.root, 'label1', id))
#                 mask2 = Image.open(os.path.join(self.root, 'label2', id))
#                 edge1 = Image.open(
#                     os.path.join(self.root, 'edge1', id))
#                 edge2 = Image.open(os.path.join(self.root, 'edge2', id))
#             else:
#                 mask1 = Image.open(os.path.join('outdir/masks/train/im1', id))
#                 mask2 = Image.open(os.path.join('outdir/masks/train/im2', id))

#             if self.mode == 'train':
#                 mask_bin = Image.open(os.path.join(self.root, 'change', id))
#                 edge_bin = Image.open(os.path.join(
#                     self.root, 'change_edge', id))
#                 # 数据增强
#                 sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
#                                          'mask_bin': mask_bin, 'edge1': edge1, 'edge2': edge2, 'edge_bin': edge_bin})
#                 img1, img2, mask1, mask2, mask_bin, edge1, edge2, edge_bin = sample['img1'], sample[
#                     'img2'], sample['mask1'], sample['mask2'], sample['mask_bin'], sample['edge1'], sample['edge2'], sample['edge_bin']
#         # print(np.array(img2).shape) (512, 512, 3)
#         img1 = self.normalize(img1)
#         img2 = self.normalize(img2)
#         mask1 = torch.from_numpy(np.array(mask1)).float()
#         mask2 = torch.from_numpy(np.array(mask2)).float()
#         mask_bin = torch.from_numpy(np.array(mask_bin)).float()
#         mask1 = mask1 / 255.0
#         mask2 = mask2 / 255.0
#         mask_bin = mask_bin / 255.0

#         edge1 = torch.from_numpy(np.array(edge1)).float()
#         edge2 = torch.from_numpy(np.array(edge2)).float()
#         edge_bin = torch.from_numpy(np.array(edge_bin)).float()
#         edge1 = edge1 / 255.0
#         edge2 = edge2 / 255.0
#         edge_bin = edge_bin / 255.0
#         # if self.mode == 'train':
#         #     return img1, img2, mask1, mask2, mask_bin, id
#         return img1, img2, mask1, mask2, mask_bin, edge1, edge2, edge_bin, id

#     def __len__(self):
#         return len(self.ids)

if __name__ == "__main__":
    # label 0-非建筑 255-建筑
    # change_gt 0-未变化 255-变化
    # img = Image.open("/home/kelin/data/train/im1/2000.png")
    # print(np.array(img).max())
    trainset = ChangeDetection(root="/home/kelin/data", mode="train")
    # kf = KFold(n_splits=3)
    # # for train_index, test_index in kf.split(trainset):
    # # print(len(train_index))
    # # print(len(test_index))
    # # valset = ChangeDetection(root="/home/kelin/data", mode="val")
    # # print(len(trainset)/len(valset))
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True,
                             pin_memory=False, num_workers=4, drop_last=True)
    # -----------------计算权重
    print(len(trainloader))
    for i in trainloader:
        print(i[2].shape)
        label_1 = i[2].numpy().astype(np.uint8)
        # 统计类别数量
        label_num1 = np.bincount(label_1.flatten())
        print(label_num1)

        label_2 = i[3].numpy().astype(np.uint8)
        label_num2 = np.bincount(label_2.flatten())
        print(label_num2)

        label_bin = i[4].numpy().astype(np.uint8)
        label_num_bin = np.bincount(label_bin.flatten())
        print(label_num_bin)

        edge1 = i[5].numpy().astype(np.uint8)
        edge1 = np.bincount(edge1.flatten())
        print(edge1)

        edge2 = i[6].numpy().astype(np.uint8)
        edge2 = np.bincount(edge2.flatten())
        print(edge2)

        edge_bin = i[7].numpy().astype(np.uint8)
        edge_bin = np.bincount(edge_bin.flatten())
        print(edge_bin)

    weights1 = label_num1.sum() / (label_num1 * 2)
    print(weights1)
    weights2 = label_num2.sum() / (label_num2 * 2)
    print(weights2)
    weights_bin = label_num_bin.sum() / (label_num_bin * 2)
    print(weights_bin)
    weights_edge1 = edge1.sum() / (edge1 * 2)
    weights_edge2 = edge2.sum() / (edge2 * 2)
    print(weights_edge1)
    print(weights_edge2)
    weights_edge_bin = edge_bin.sum() / (edge_bin * 2)
    print(weights_edge_bin)
    # # break
    # label1 = pd.DataFrame([481903295, 42384705], columns=['t1_label'])
    # label2 = pd.DataFrame([477472476, 46815524])
    # label_bin = pd.DataFrame([513696736, 10591264])
    # label1["t2_label"] = label2[0]
    # label1["change_label"] = label_bin[0]
    # label1.plot(kind='bar')
    # plt.savefig("label.png")
    # print(label1)
    # *********************计算均值和方差**************************
    # dataset = ChangeDetection(root="/home/kelin/data", mode="train")
    # print(len(dataset))
    # trainloader = DataLoader(dataset, batch_size=1, shuffle=True,
    #                          pin_memory=False, num_workers=4, drop_last=False)
    # for data in trainloader:
    #     print(data[0].shape)
    #     break
    # print("img1 mean", [data[0][:, i, :, :].mean() for i in range(3)])
    # print("img1 std", [data[0][:, i, :, :].std() for i in range(3)])
    # print("img2 mean", [data[1][:, i, :, :].mean() for i in range(3)])
    # print("img2 std", [data[1][:, i, :, :].std() for i in range(3)])
