from datasets.change_detection import ChangeDetection
from datasets.segmentation import Segmentation
from models.model_zoo import get_model
from utils.image import visualize_CD
from utils.palette import color_map
import numpy as np
import os
from PIL import Image
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import json
from torchgeo1.models import FarSeg_CD, FarSeg, FarSeg_CD_Res
from torchgeo.models import ChangeStarFarSeg, ChangeModel
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.nn import DataParallel


class Dict(dict):
    # # self.属性写入 等价于调用dict.__setitem__
    __setattr__ = dict.__setitem__
    # # self.属性读取 等价于调用dict.__setitem__
    __getattribute__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


if __name__ == "__main__":
    # 开始时间
    START_TIME = time.time()
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.is_available())
    # 加载数据
    testset_bin = ChangeDetection(root="datasets/data/val_0", mode="val")
    # testset_bin = binmentation(root="/home/kelin/data/train", mode="val")

    testloader = DataLoader(testset_bin, batch_size=1, shuffle=True,
                            pin_memory=True, num_workers=0, drop_last=False)
    # 加载模型
    # model1 = ChangeStarFarSeg(
    #     backbone="resnext50_32x4d", backbone_pretrained=False)
    model2 = ChangeModel(
        backbone="hrnet_w30", backbone_pretrained=False)
    # model3 = FarSeg_CD_Res("resnet50", classes=2,
    #                        backbone_pretrained=False)

    # model1 = DataParallel(model1)
    model2 = DataParallel(model2)
    # model3 = DataParallel(model3)

    # 加载权重文件
    # model1_pth = torch.load(
    #     'exp_result_original_data/ChangeModel_resnext50_32x4d_change_v2_2021-10-15-00:09/checkpoints/ChangeModel_resnext50_32x4d_bin_45.18.pth')
    model2_pth = torch.load(
        'exp_result_original_data/ChangeModel_hrnet_w30_fold0_2021-10-15-19:32/checkpoints/ChangeModel_hrnet_w30_bin_40.74.pth')
    # model3_pth = torch.load(
    #     'test_model/farseg_resnet50_bin_50.03.pth')

    # model_bin.load_state_dict(model_change_pth, strict=True)
    # model1.load_state_dict(model1_pth, strict=True)
    model2.load_state_dict(model2_pth, strict=True)
    # model3.load_state_dict(model3_pth, strict=True)

    models = [model2]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()
    cmap = color_map()
    tbar = tqdm(testloader)
    # 开始测试
    # print(MODE)
    from utils.metric import F1_score
    if not os.path.exists('./output_path'):
        os.mkdir('./output_path')
    metric_bin = F1_score(num_classes=2)
    with torch.no_grad():
        for k, (img1, img2, mask_bin, id) in enumerate(tbar):
            img1, img2 = img1.cuda(
                non_blocking=True), img2.cuda(non_blocking=True)
            img = torch.cat([torch.unsqueeze(img1, dim=1),
                             torch.unsqueeze(img2, dim=1)], dim=1)
            img1_ = img.flip(3)
            img2_ = img.flip(4)
            img3_ = img.transpose(3, 4)
            out_list = []
            # for model in models:
            # out1 = model1(img1, img2)
            # out1 = torch.unsqueeze(out1[:, 1, :, :], dim=1)

            # out1 = model1(img)['change_prob']
            out2 = model2(img)['change_prob']
            out_list.append(out2)

            out_cd1 = model2(img1_)['change_prob']
            out_list.append(out_cd1.flip(2))

            out_cd2 = model2(img2_)['change_prob']
            out_list.append(out_cd2.flip(3))

            out_cd3 = model2(img3_)['change_prob']
            out_list.append(out_cd3.transpose(2, 3))

            # out3 = model3(img1, img2)
            # out3 = torch.unsqueeze(out3[:, 1, :, :], dim=1)

            # # out_list.append(out1)
            # out_list.append(out2)
            # # out_list.append(out3)
            # # # 将不同模型的预测结果合并
            out = torch.stack(out_list, dim=0)
            # # # 计算不同模型预测结果的均值
            out = torch.sum(out, dim=0) / 4
            # # # (bz,h,w)
            out = np.uint8((out > 0.5).cpu().numpy())
            # 计算预测精度
            metric_bin.add_batch(out, mask_bin.numpy())
            # if k % 50 == 0:
            #     # 可视化
            #     for i in range(1):
            #         visualize_CD([img1[i].cpu().numpy(), img2[i].cpu().numpy()],
            #                      [mask_bin[i].cpu().numpy(), out1[i]],
            #                      './output_path', id[i] + ".png")

        # 计算F1-Score
        f1_bin = metric_bin.evaluate()
        Score = 0.6*f1_bin
        print(
            "f1_bin:{}  Score:{}".format(f1_bin, Score))
    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))
