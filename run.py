
import os
import sys

from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_data import Test_CD
from torchgeo1.models import FarSeg, FarSeg_CD
from torchgeo.models.changestar import ChangeModel
from utils.palette import color_map

if __name__ == "__main__":
    # 测试是否GPU可用
    print(torch.cuda.is_available())
    # 加载数据
    testset = Test_CD(root=sys.argv[1])
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)
    # 语义分割模型
    model_seg = FarSeg(backbone="hrnet_w30", classes=2,
                       backbone_pretrained=False)
    # model_seg = DataParallel(model_seg)
    # model_seg_pth = torch.load('seg_models/farseg_resnet50_seg_8.03.pth')
    # model_seg.load_state_dict(model_seg_pth, strict=True)

    # 变化检测模型
    model_cd = ChangeModel(backbone="hrnet_w30", backbone_pretrained=False)
    model_cd = DataParallel(model_cd)
    model_si = FarSeg_CD(backbone="hrnet_w30", classes=2,
                         backbone_pretrained=False)
    model_si = DataParallel(model_si)
    
    # 设置模型为测试模式
    models = [model_cd, model_seg, model_si]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    # 获取模型权重
    seg_pths = glob("models_seg/*.pth")
    cd_pths = glob("models_cd/*.pth")
    si_pths = glob("models_si/*.pth")
    
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    cmap = color_map()
    # 开始测试
    with torch.no_grad():
        for img1, img2, id in tqdm(testloader):
            img1, img2 = img1.cuda(
                non_blocking=True), img2.cuda(non_blocking=True)
            img = torch.cat([torch.unsqueeze(img1, dim=1),
                             torch.unsqueeze(img2, dim=1)], dim=1)
            
            # 1.获取建筑物提取结果
            out_seg1_list = []
            out_seg2_list = []
            for pth in seg_pths:
                model_seg.load_state_dict(torch.load(pth))
                out_seg1 = model_seg(img1, True)
                out_seg2 = model_seg(img2, True)
                out_seg1_list.append(out_seg1)
                out_seg2_list.append(out_seg2)

            # 将不同的变化检测模型的预测结果合并
            out_seg1 = torch.stack(out_seg1_list, dim=0)
            # 计算不同模型预测结果的均值
            out_seg1 = torch.sum(out_seg1, dim=0) / len(seg_pths)
            # 转换成one-hot
            out_seg1 = torch.argmax(out_seg1, dim=1).cpu().numpy().squeeze()

            # 将不同的变化检测模型的预测结果合并
            out_seg2 = torch.stack(out_seg2_list, dim=0)
            # 计算不同模型预测结果的均值
            out_seg2 = torch.sum(out_seg2, dim=0) / len(seg_pths)
            # 转换成one-hot
            out_seg2 = torch.argmax(out_seg2, dim=1).cpu().numpy().squeeze()

            # 2.获取变化检测结果
            out_cd_list = []
            out_si_list = []
            
            for pth in cd_pths:
                model_cd.load_state_dict(torch.load(pth))
                out_cd = model_cd(img)['change_prob']
                out_cd_list.append(out_cd)
                # out_cd1 = model_cd(img.flip(3))['change_prob']
                # out_cd_list.append(out_cd1.flip(2))

                # out_cd2 = model_cd(img.flip(4))['change_prob']
                # out_cd_list.append(out_cd2.flip(3))

                # out_cd3 = model_cd(img.transpose(3, 4))['change_prob']
                # out_cd_list.append(out_cd3.transpose(2, 3))
                
            for pth in si_pths:
                model_si.load_state_dict(torch.load(pth))
                out_si = model_si(img1, img2)
                out_si = torch.unsqueeze(out_si[:, 1, :, :], dim=1)
                out_si_list.append(out_si)
                
            out_list = out_cd_list + out_si_list
            # 将不同的变化检测模型的预测结果合并
            out_cd = torch.stack(out_list, dim=0)
            # 计算不同模型预测结果的均值
            out_cd = torch.sum(out_cd, dim=0) / (len(cd_pths)+len(si_pths))
            # 转换成one-hot
            out_cd = (out_cd > 0.5).cpu().numpy().squeeze()

            # 保存预测结果
            mask1 = Image.fromarray(out_seg1.astype(np.uint8), mode="P")
            mask1.putpalette(cmap)
            mask1.save(sys.argv[2] + "/" + id[0]+"_1_label.png")

            mask2 = Image.fromarray(out_seg2.astype(np.uint8), mode="P")
            mask2.putpalette(cmap)
            mask2.save(sys.argv[2] + "/" + id[0]+"_2_label.png")

            mask_cd = Image.fromarray(out_cd.astype(np.uint8), mode="P")
            mask_cd.putpalette(cmap)
            mask_cd.save(sys.argv[2] + "/" + id[0]+"_change.png")
