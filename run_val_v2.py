
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_data import Test_CD
from torchgeo1.models import FarSeg
from torchgeo.models.changestar import ChangeModel
from utils.palette import color_map

if __name__ == "__main__":
    # 加载数据
    testset = Test_CD(root=sys.argv[1])
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)

    # 语义分割模型
    model_seg = FarSeg(backbone="resnet50", classes=2,
                       backbone_pretrained=False)
    model_seg_pth = torch.load('seg_models/farseg_resnet50_seg_8.03.pth')
    model_seg.load_state_dict(model_seg_pth, strict=True)

    # 变化检测模型
    model_cd = ChangeModel(backbone="hrnet_w30", backbone_pretrained=False)
    model_cd = DataParallel(model_cd)

    # 设置模型为测试模式
    models = [model_cd, model_seg]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    # 获取模型权重
    pths = os.listdir("fold_models")
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    # 开始测试
    with torch.no_grad():
        for img1, img2, id in tqdm(testloader):
            img1, img2 = img1.cuda(
                non_blocking=True), img2.cuda(non_blocking=True)
            img = torch.cat([torch.unsqueeze(img1, dim=1),
                             torch.unsqueeze(img2, dim=1)], dim=1)

            # 1.获取建筑物提取结果
            out1 = model_seg(img1, True)
            out2 = model_seg(img2, True)
            out1 = torch.argmax(out1, dim=1).cpu().numpy().squeeze()
            out2 = torch.argmax(out2, dim=1).cpu().numpy().squeeze()

            # 2.获取变化检测结果
            out_cd_list = []
            for pth in pths:
                model_cd.load_state_dict(torch.load(pth))
                out_cd = model_cd(img)['change_prob']
                out_cd_list.append(out_cd)
            # 将不同的变化检测模型的预测结果合并
            out_cd = torch.stack(out_cd_list, dim=0)
            # 计算不同模型预测结果的均值
            out_cd = torch.sum(out_cd, dim=0) / len(pths)
            # 转换成one-hot
            out_cd = (out_cd > 0.5).cpu().numpy().squeeze()

            # 保存预测结果
            mask1 = Image.fromarray(out1.astype(np.uint8), mode="L")
            mask1.save(sys.argv[2] + "/" + id+"_1_label.png")
            mask2 = Image.fromarray(out2.astype(np.uint8), mode="L")
            # mask2.putpalette(cmap)
            mask2.save(sys.argv[2] + "/" + id+"_2_label.png")

            mask_cd = Image.fromarray(out_cd.astype(np.uint8), mode="L")
            # mask_bin.putpalette(cmap)
            mask_cd.save(sys.argv[2] + "/" + id+"_change.png")
