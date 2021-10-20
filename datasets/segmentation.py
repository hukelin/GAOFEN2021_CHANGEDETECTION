from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
import datasets.transform as tr
# import transform as tr
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# from .multi_scale import MultiScale, RandomFlip
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, PadIfNeeded
import ever as er

# class Segmentation(Dataset):
#     CLASSES = ['未变化区域', "非建筑物", '建筑物']

#     def __init__(self, root, mode):
#         super(Segmentation, self).__init__()
#         self.root = root
#         self.mode = mode
#         if mode in ['train', 'val', 'val_im']:
#             self.ids = os.listdir(os.path.join(self.root, "im2"))
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
#         # 转为tensor
#         self.normalize = transforms.Compose([
#             transforms.ToTensor()  # PIL 读取的图像为[512,512,3]
#             # 这里的 ToTensor 会有转换维度的操作
#         ])
#         # 数据增强：随机的翻转和旋转
#         self.transform = transforms.Compose([
#             tr.RandomFlipOrRotate()
#         ])
#         # # 中心裁剪
#         # self.fiveCrop = transforms.Compose([
#         #     transforms.FiveCrop(512)
#         # ])
#         # # self.randomCrop = transforms.RandomCrop(512)
#         # # self.centerCrop = transforms.CenterCrop(512)

#     def __getitem__(self, index):
#         id = self.ids[index]
#         if self.mode in ['train', 'val']:
#             img = Image.open(os.path.join(self.root, 'im2', id))
#             label = Image.open(os.path.join(self.root, 'label2', id))
#         if self.mode == "val_im":
#             img = Image.open(os.path.join(self.root, 'im1', id))
#             label = Image.open(os.path.join(self.root, 'label1', id))
#         # 读取测试数据
#         if self.mode == "test":
#             # 读取时相1的图像
#             img1 = Image.open(os.path.join(self.root, id+"_1.png"))
#             # 读取时相2的图像
#             img2 = Image.open(os.path.join(self.root, id+"_2.png"))
#             img1 = self.normalize(img1)
#             img2 = self.normalize(img2)
#             return img1, img2, id

#         if self.mode == 'train':
#             # 数据增强
#             sample = self.transform({'img': img, 'label': label})
#             img, label = sample['img'], sample['label']
#         # numpy转tensor
#         img = self.normalize(img)
#         label = torch.from_numpy(np.array(label)).float()
#         label = label / 255.0
#         return img, label, id.split('.')[0]

#     def __len__(self):
#         return len(self.ids)
from glob import glob


class Segmentation(Dataset):
    CLASSES = ['未变化区域', "非建筑物", '建筑物']

    def __init__(self, root, mode):
        super(Segmentation, self).__init__()
        self.root = root
        self.mode = mode

        im1 = os.path.join(self.root, "im1")
        im2 = os.path.join(self.root, "im2")
        self.ids = glob(im1+"/*.png") + glob(im2+"/*.png")
        
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

    def __getitem__(self, index):
        id = self.ids[index]
        name = os.path.basename(id).split('.')[0]
        img = np.array(Image.open(id))
        if "im1" in id:
            label_id = id.replace("im1", "label1")
            label = np.array(Image.open(label_id))
        else:
            label_id = id.replace("im2", "label2")
            label = np.array(Image.open(label_id))
        if self.mode == 'train':
            # 数据增强
            blob = self.transforms_t(**dict(image=img, mask=label))
            img = blob['image'].float()
            label = blob['mask'].float()
            if "im1" in id:
                img = self.normalize1(img)
            else:
                img = self.normalize2(img)
            label = label / 255.
            return img, label, name
        if self.mode == 'val':
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            if "im1" in id:
                img = self.normalize1(img)
            else:
                img = self.normalize2(img)
            label = label / 255.
            return img, label, name

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    # label 0-非建筑 255-建筑
    # change_gt 0-未变化 255-变化
    # img = Image.open("/home/kelin/data/train/im1/2000.png")
    # print(np.array(img).max())
    # trainset = Segmentation(
    #     root="/home/kelin/code/GaoFen2021_Segmentation/datasets", mode="train")
    # a = trainset.__getitem__(0)
    # print(a)
    # LEVIR_CD = "/home/kelin/code/GaoFen2021_Segmentation/datasets/LEVIR_CD.npz"
    # GF_CD = "/home/kelin/code/GaoFen2021_Segmentation/datasets/data_gf.npz"
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
    #         "/home/kelin/code/GaoFen2021_Segmentation/datasets/data_CD", f)
    #     np.savez(npz_f, img1=img1[i], img2=img2[i], label=label[i])
    trainset = Segmentation(
        root="/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_seg", mode="train")
    # print(len(testset))
    # print(testset[2][1].max())

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
    # valset = Segmentation(root="/home/kelin/data/data_CD", mode="val")
    # print(len(valset))
    # kf = KFold(n_splits=3)
    # # for train_index, test_index in kf.split(trainset):
    # # print(len(train_index))
    # # print(len(test_index))
    # # valset = Segmentation(root="/home/kelin/data", mode="val")
    # # print(len(trainset)/len(valset))
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True,
                             pin_memory=False, num_workers=4, drop_last=True)
    # for im, la, id in tqdm(trainset):
    #     f = "/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_seg"
    #     np.savez(os.path.join(f, id+".npz"), img=im, label=la)
    # s = np.load("/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_CD/0.npz",allow_pickle=True)
    # print(s['img1'].shape)
    # print(s['label'].shape)
    # # # -----------------计算权重
    # # print(len(trainloader))
    for i in trainloader:
        # [torch.Size([2109, 3, 512, 512]), torch.Size([2109, 3, 512, 512]), torch.Size([2109, 512, 512])]
        # print(i[0].shape)
        # break
        label_1 = i[1].numpy().astype(np.uint8)
        # 统计类别数量
        label_num1 = np.bincount(label_1.flatten())
        print(label_num1)
    weights1 = label_num1.sum() / (label_num1 * 2)
    print(weights1)
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
    # dataset = Segmentation(root="/home/kelin/data", mode="train")
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
