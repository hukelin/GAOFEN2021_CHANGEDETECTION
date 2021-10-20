from torch.utils.data import DataLoader
# import datasets.transform as tr
# import transform as tr
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations.augmentations.functional as F
from sklearn.model_selection import train_test_split
# from .multi_scale import MultiScale, RandomFlip
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, PadIfNeeded
import ever as er

# 添加K-fold交叉验证


class ChangeDetection(Dataset):
    def __init__(self, root, mode):
        super(ChangeDetection, self).__init__()
        self.root = root
        self.mode = mode
        # if mode in ['train', 'val']:
        #     # 训练数据集文件夹
        #     self.root = os.path.join(self.root, 'train')
        #     # 读取训练图像
        #     self.ids = os.listdir(os.path.join(self.root, "im1"))
        #     # 划分数据集-训练集:验证集
        #     self.train_ids, self.val_ids = train_test_split(
        #         self.ids, test_size=0.25, random_state=0)
        #     if mode == 'train':
        #         self.ids = self.train_ids
        #     else:
        #         self.ids = self.val_ids
        self.ids = os.listdir(os.path.join(self.root, "im1"))
        # else:
        #     # 训练数据集文件夹
        #     self.root = os.path.join(self.root, 'test_AB')
        #     # 读取训练图像
        #     self.ids = os.listdir(os.path.join(self.root, "im1"))
        # self.ids.sort()
        # 数据增强：随机的翻转和旋转
        # self.transform_Img = transforms.Compose([
        #     tr.RandomFlipOrRotate(),
        # ])
        # 全部数据
        self.normalize1 = transforms.Compose([
            transforms.Normalize(mean=(90.3236, 89.1732, 80.8296),
                                 std=(47.2191, 40.7412, 41.1059))
        ])
        self.normalize2 = transforms.Compose([
            transforms.Normalize(mean=(80.4520, 81.5796, 74.7567),
                                 std=(50.5237, 45.2135, 48.5634))
        ])

        self.transforms_t = Compose([
            OneOf([
                HorizontalFlip(True),
                VerticalFlip(True),
                RandomRotate90(True)
            ], p=0.5),
            er.preprocess.albu.RandomDiscreteScale(
                [1.25, 1.5], p=0.5),
            # PadIfNeeded(512, 512),
            RandomCrop(512, 512, True),
            er.preprocess.albu.ToTensor()
        ])

        # self.transforms_v = Compose([
        #     RandomCrop(512, 512, True),
        #     er.preprocess.albu.ToTensor()
        # ])
        # # 中心裁剪
        # # self.randomCrop = transforms.RandomCrop(512)
        # # self.centerCrop = transforms.CenterCrop(512)

    def __getitem__(self, index):
        id = self.ids[index]
        name = id.split('.')[0]
        if self.mode == "train":
            # img1 = Image.open(os.path.join(self.root, 'im1', id))
            # img2 = Image.open(os.path.join(self.root, 'im2', id))
            # label = Image.open(os.path.join(self.root, 'label', name+".png"))
            img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
            img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
            label = np.array(Image.open(os.path.join(self.root, 'label', id)))
            # # 数据增强
            # sample = self.transform_Img(
            #     {'img1': img1, 'img2': img2, 'label': label})
            # img1, img2, label = sample['img1'], sample['img2'], sample['label']
            imgs = np.concatenate([img1, img2], axis=2)
            blob = self.transforms_t(**dict(image=imgs, mask=label))
            imgs = blob['image']
            label = blob['mask'].float()
            img1 = imgs[:3, :, :].float()
            img2 = imgs[3:, :, :].float()
            img1 = self.normalize1(img1)
            img2 = self.normalize2(img2)
            label = label / 255.
            return img1, img2, label, name
        if self.mode == 'val':
            img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
            img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
            label = np.array(Image.open(os.path.join(self.root, 'label', id)))
            img1 = img1.transpose((2, 0, 1))
            img2 = img2.transpose((2, 0, 1))
            # numpy转tensor
            img1 = torch.from_numpy(np.array(img1)).float()
            img2 = torch.from_numpy(np.array(img2)).float()
            label = torch.from_numpy(np.array(label)).float()
            img1 = self.normalize1(img1)
            img2 = self.normalize2(img2)
            # imgs = np.concatenate([img1, img2], axis=2)
            # blob = self.transforms_v(**dict(image=imgs, mask=label))
            # imgs = blob['image']
            # label = blob['mask'].float()
            # img1 = imgs[:3, :, :].float()
            # img2 = imgs[3:, :, :].float()
            label = label / 255.
            return img1, img2, label, name
        # 测试数据
        if self.mode == "test":
            img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
            img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
            img1 = img1.transpose((2, 0, 1))
            img2 = img2.transpose((2, 0, 1))
            # numpy转tensor
            img1 = torch.from_numpy(np.array(img1)).float()
            img2 = torch.from_numpy(np.array(img2)).float()
            return img1, img2, name
            # imgs = np.concatenate([img1, img2], axis=2)
            # blob = self.transforms(**dict(image=imgs))
            # imgs = blob['image']
            # img1 = imgs[:3, :, :].float()
            # img2 = imgs[3:, :, :].float()
            # return img1, img2, name
        # 数据增强
        # if self.mode == "train":
        #     # imgs = np.concatenate([img1, img2], axis=2)
        #     # blob = self.transforms(**dict(image=imgs, mask=label))
        #     # imgs = blob['image']
        #     # label = blob['mask'].float()
        #     # img1 = imgs[:3, :, :].float()
        #     # img2 = imgs[3:, :, :].float()
        #     # return img1, img2, label, name
        #     # img1 = img1.transpose((2, 0, 1))
        #     # img2 = img2.transpose((2, 0, 1))
        #     gt_mask1 = np.array(Image.open(
        #         os.path.join(self.root, 'label1', id)))
        #     label = np.zeros_like(gt_mask1)
        #     # 原始标签图中未变化区域的值为0
        #     # 二值分类中，未变化为1，变化为0
        #     label[gt_mask1 == 0] = 1
        #     label = Image.fromarray(label)

        #     # numpy转tensor
        #     img1 = torch.from_numpy(np.array(img1)).float()
        #     img2 = torch.from_numpy(np.array(img2)).float()
        #     label = torch.from_numpy(np.array(label)).float()
        #     return img1, img2, label, name
        # if self.mode == "val":
        #     img1 = img1.transpose((2, 0, 1))
        #     img2 = img2.transpose((2, 0, 1))
        #     # numpy转tensor
        #     img1 = torch.from_numpy(np.array(img1)).float()
        #     img2 = torch.from_numpy(np.array(img2)).float()
        #     label = torch.from_numpy(np.array(label)).float()
        #     return img1, img2, label, name

    def __len__(self):
        return len(self.ids)
# class ChangeDetection(Dataset):

#     def __init__(self, root, mode):
#         super(ChangeDetection, self).__init__()
#         self.root = root
#         self.mode = mode
#         if mode in ['train', 'val']:
#             # 训练数据集文件夹
#             self.root = os.path.join(self.root, 'train')
#             # 读取训练图像
#             self.ids = os.listdir(os.path.join(self.root, "im1"))
#             # 训练图像排序
#             self.ids.sort()
#             # 划分数据集-训练集:验证集 = 4:1
#             self.train_ids, self.val_ids = train_test_split(
#                 self.ids, test_size=0.25, random_state=0)
#             if mode == 'train':
#                 self.ids = self.train_ids
#             else:
#                 self.ids = self.val_ids
#         else:
#             self.root = os.path.join(self.root)
#             self.ids = os.listdir(self.root)
#         self.ids.sort()
#         self.transforms = Compose([
#             OneOf([
#                 HorizontalFlip(True),
#                 VerticalFlip(True),
#                 RandomRotate90(True)
#             ], p=0.5),
#             er.preprocess.albu.RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
#             PadIfNeeded(512, 512),
#             RandomCrop(512, 512, True),
#             er.preprocess.albu.ToTensor()])
#         # # 数据增强：随机的翻转和旋转
#         # self.transform = transforms.Compose([
#         #     tr.RandomFlipOrRotate(),
#         # ])

#         # # 数据标准化
#         # self.normalize = transforms.Compose([
#         #     # transforms.ColorJitter(
#         #     #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#         #     transforms.ToTensor()
#         #     # transforms.Normalize(
#         #     #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         # ])
#         # # 中心裁剪
#         # self.fiveCrop = transforms.Compose([
#         #     transforms.FiveCrop(512)
#         # ])
#         # self.randomCrop = transforms.RandomCrop(512)
#         # self.centerCrop = transforms.CenterCrop(512)

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
#             mask_bin = Image.open(os.path.join(self.root, 'label', id))
#         else:
#             if self.mode == 'train':
#                 mask_bin = Image.open(os.path.join(self.root, 'label', id))
#                 # # 数据增强
#                 # sample = self.transform(
#                 #     {'img1': img1, 'img2': img2, 'mask_bin': mask_bin})
#                 # img1, img2, mask_bin = sample['img1'], sample['img2'], sample['mask_bin']
#         img1 = self.normalize(img1)
#         img2 = self.normalize(img2)
#         mask_bin = torch.from_numpy(np.array(mask_bin)).float()
#         mask_bin = mask_bin / 255.0
#         # if img1.shape[1:] == (1024, 1024):
#         #     img1 = self.fiveCrop(img1)
#         #     img2 = self.fiveCrop(img2)
#         #     mask_bin = self.fiveCrop(mask_bin)
#         #     img1 = torch.stack([i for i in img1])
#         #     img2 = torch.stack([i for i in img2])
#         #     mask_bin = torch.stack([i for i in mask_bin])
#         # return img1, img2, mask_bin, id
#         return img1, img2, mask_bin, id

#     def __len__(self):
#         return len(self.ids)


# class ChangeDetection(Dataset):
#     # CLASSES = ['未变化区域', "非建筑物", '建筑物']

#     def __init__(self, root, mode):
#         super(ChangeDetection, self).__init__()
#         self.root = root
#         self.mode = mode
#         if mode in ['train', 'val']:
#             self.ids = os.listdir(self.root)
#             # 划分数据集-训练集:验证集
#             self.train_ids, self.val_ids = train_test_split(
#                 self.ids, test_size=0.3, random_state=0)
#             if mode == 'train':
#                 self.ids = self.train_ids
#             else:
#                 self.ids = self.val_ids
#         else:
#             # 获取图像id
#             self.ids = [str(i)
#                         for i in range(1, int(len(os.listdir(self.root)) / 2)+1)]
#         self.ids.sort()
#         # # 数据增强：随机的翻转和旋转
#         # 转为tensor
#         self.normalize = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         self.transforms = Compose([
#             OneOf([
#                 HorizontalFlip(True),
#                 VerticalFlip(True),
#                 RandomRotate90(True)
#             ], p=0.5),
#             er.preprocess.albu.RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
#             PadIfNeeded(512, 512),
#             RandomCrop(512, 512, True),
#             er.preprocess.albu.ToTensor()
#         ])
#         # # 中心裁剪
#         # # self.randomCrop = transforms.RandomCrop(512)
#         # # self.centerCrop = transforms.CenterCrop(512)

#     def __getitem__(self, index):
#         id = self.ids[index]
#         if self.mode in ['train', 'val']:
#             file = os.path.join(self.root, id)
#             sample = np.load(file, allow_pickle=True)
#             img1 = sample['img1']
#             img2 = sample['img2']
#             label = sample['label']

#         # 测试数据
#         if self.mode == "test":
#             # 读取时相1的图像
#             img1 = Image.open(os.path.join(self.root, id+"_1.png"))
#             # 读取时相2的图像
#             img2 = Image.open(os.path.join(self.root, id+"_2.png"))
#             img1 = self.normalize(img1)
#             img2 = self.normalize(img2)
#             return img1, img2, id
#         # 数据增强
#         if self.mode == "train":
#             img1 = img1.transpose((1, 2, 0))
#             img2 = img2.transpose((1, 2, 0))
#             imgs = np.concatenate([img1, img2], axis=2)
#             blob = self.transforms(**dict(image=imgs, mask=label))
#             imgs = blob['image']
#             label = blob['mask']
#             img1 = imgs[:3, :, :]
#             img2 = imgs[3:, :, :]
#             return img1, img2, label, id.split('.')[0]
#         # numpy转tensor
#         img1 = torch.from_numpy(np.array(img1)).float()
#         img2 = torch.from_numpy(np.array(img2)).float()
#         label = torch.from_numpy(np.array(label)).float()

#         return img1, img2, label, id.split('.')[0]

#     def __len__(self):
#         return len(self.ids)


if __name__ == "__main__":
    # label 0-非建筑 255-建筑
    # change_gt 0-未变化 255-变化
    # img = Image.open("/home/kelin/data/train/im1/2000.png")
    # print(np.array(img).max())
    # trainset = ChangeDetection(
    #     root="/home/kelin/code/GaoFen2021_ChangeDetection/datasets", mode="train")
    # im1 = trainset.__getitem__(0)
    # print(im1)
    # LEVIR_CD = "/home/kelin/code/GaoFen2021_ChangeDetection/datasets/LEVIR_CD.npz"
    # GF_CD = "/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_gf.npz"
    # LEVIR_file = np.load(LEVIR_CD, allow_pickle=True)
    # GF_file = np.load(GF_CD, allow_pickle=True)
    # img1 = np.concatenate([LEVIR_file['img1'], GF_file['img1']])
    # # print(self.img1.shape) # (5185, 3, 512, 512)
    # img2 = np.concatenate([LEVIR_file['img2'], GF_file['img2']])
    # label = np.concatenate(
    #     [LEVIR_file['label'], GF_file['label']])
    # for i in range(len(img1)):
    #     f = str(i) + ".npz"
    #     npz_f = os.path.join(
    #         "/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_CD", f)
    #     np.savez(npz_f, img1=img1[i], img2=img2[i], label=label[i])
    # trainset = ChangeDetection(
    #     root="/home/kelin/data/GaoFen", mode="val")
    # print(len(testset))
    # print(testset[0][0].shape)
    # for i in testset:
    #     print()
    # print(LEVIR_file['img1'][0].shape)
    # ids = [i for i in range(len(LEVIR_file['img1']))]
    # print(ids)
    # # 划分数据集-训练集:验证集
    # train_ids, val_ids = train_test_split(
    # ids, test_size=0.4, random_state=0)
    # print(train_ids)
    # print(GF_file['img1'].shape)
    # print(LEVIR_file['img2'].shape)
    # print(GF_file['img2'].shape)
    # print(LEVIR_file['label'].shape)
    # print(GF_file['label'].shape)
    # GF_file = np.load(GF_CD, allow_pickle=True)
    # valset = ChangeDetection(root="/home/kelin/data/data_CD", mode="val")
    # print(len(valset))
    # kf = KFold(n_splits=3)
    # # for train_index, test_index in kf.split(trainset):
    # # print(len(train_index))
    # # print(len(test_index))
    # # valset = ChangeDetection(root="/home/kelin/data", mode="val")
    # # print(len(trainset)/len(valset))
    # trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True,
    #                          pin_memory=False, num_workers=4, drop_last=True)
    # # -----------------计算权重
    # print(len(trainloader))
    # for i in trainloader:
    # [torch.Size([2109, 3, 512, 512]), torch.Size([2109, 3, 512, 512]), torch.Size([2109, 512, 512])]
    # print(i[0].shape)
    # break
    # label_1 = i[2].numpy().astype(np.uint8)
    # # 统计类别数量
    # label_num1 = np.bincount(label_1.flatten())
    # print(label_num1)
    # label_2 = i[3].numpy().astype(np.uint8)
    # label_num2 = np.bincount(label_2.flatten())
    # print(label_num2)

    # label_bin = i[4].numpy().astype(np.uint8)
    # label_num_bin = np.bincount(label_bin.flatten())
    # print(label_num_bin)

    # edge1 = i[5].numpy().astype(np.uint8)
    # edge1 = np.bincount(edge1.flatten())
    # print(edge1)

    # edge2 = i[6].numpy().astype(np.uint8)
    # edge2 = np.bincount(edge2.flatten())
    # print(edge2)

    # edge_bin = i[7].numpy().astype(np.uint8)
    # edge_bin = np.bincount(edge_bin.flatten())
    # print(edge_bin)
    # label_num1 = np.array([795989887+513696736, 38938753+10591264])
    # weights1 = label_num1.sum() / (label_num1 * 2)
    # print(weights1)
    # weights2 = label_num2.sum() / (label_num2 * 2)
    # print(weights2)
    # weights_bin = label_num_bin.sum() / (label_num_bin * 2)
    # print(weights_bin)
    # weights_edge1 = edge1.sum() / (edge1 * 2)
    # weights_edge2 = edge2.sum() / (edge2 * 2)
    # print(weights_edge1)
    # print(weights_edge2)
    # weights_edge_bin = edge_bin.sum() / (edge_bin * 2)
    # print(weights_edge_bin)
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
    dataset = ChangeDetection(root="/home/kelin/data/train", mode="train")
    print(len(dataset))
    trainloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True,
                             pin_memory=False, num_workers=4, drop_last=False)
    for data in trainloader:
        print(data[0].shape)
        # break
        print("img1 mean", [data[0][:, i, :, :].mean() for i in range(3)])
        print("img1 std", [data[0][:, i, :, :].std() for i in range(3)])
        print("img2 mean", [data[1][:, i, :, :].mean() for i in range(3)])
        print("img2 std", [data[1][:, i, :, :].std() for i in range(3)])
