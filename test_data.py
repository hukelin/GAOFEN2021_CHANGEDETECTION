import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


class Test_CD(Dataset):

    def __init__(self, root):
        super(Test_CD, self).__init__()
        self.root = root
        # 获取图像id
        self.ids = [str(i)
                    for i in range(1, int(len(os.listdir(self.root)) / 2)+1)]
        self.ids.sort()
        self.normalize1 = transforms.Compose([
            transforms.Normalize(mean=(90.3236, 89.1732, 80.8296),
                                 std=(47.2191, 40.7412, 41.1059))
        ])
        self.normalize2 = transforms.Compose([
            transforms.Normalize(mean=(80.4520, 81.5796, 74.7567),
                                 std=(50.5237, 45.2135, 48.5634))
        ])

    def __getitem__(self, index):
        id = self.ids[index]
        # 读取时相1的图像
        img1 = Image.open(os.path.join(self.root, id+"_1.png"))
        # 读取时相2的图像
        img2 = Image.open(os.path.join(self.root, id+"_2.png"))

        img1 = np.array(img1).transpose((2, 0, 1))
        img2 = np.array(img2).transpose((2, 0, 1))

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        img1 = self.normalize1(img1)
        img2 = self.normalize2(img2)
        return img1, img2, id

    def __len__(self):
        return len(self.ids)
