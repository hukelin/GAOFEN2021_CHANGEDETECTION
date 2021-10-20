
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
from datasets.change_detection import LChangeDetection
from torchgeo1.models import FarSeg, FarSeg_CD
from torchgeo.models import ChangeStarFarSeg
from torchgeo.models.changestar import ChangeModel
from utils.palette import color_map

if __name__ == "__main__":
    # 开始时间
    START_TIME = time.time()
    torch.backends.cudnn.benchmark = True
    # 测试是否GPU可用
    print(torch.cuda.is_available())
    # 加载数据
    # testset = Test_CD(root=sys.argv[1])
    # 加载验证数据
    testset = LChangeDetection(
        root="/home/kelin/data/GaoFen", mode="val")
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)
    # 变化检测模型
    # 加载模型
    # model1 = FarSeg_CD(backbone="resnext50_32x4d", classes=2,
    #                    backbone_pretrained=False)
    # model1 = ChangeStarFarSeg(
    #     backbone="resnext50_32x4d", backbone_pretrained=False)
    # 加载模型
    model_cd2 = ChangeModel(backbone="hrnet_w30", backbone_pretrained=False)

    # model1 = DataParallel(model1)
    model_cd2 = DataParallel(model_cd2)
    # model_bin = get_model(args2.model, args2.backbone, False,
    #                    len(testset.CLASSES)-1, args2.lightweight)
    # print(model_bin)
    # 加载权重文件
    # model1_pth = torch.load(
    #     'test_model/ChangeModel_resnext50_32x4d_bin_45.49.pth')
    model2_pth = torch.load('exp_result_original_data/ChangeModel_hrnet_w30_fold0_2021-10-15-19:32/checkpoints/ChangeModel_hrnet_w30_bin_40.74.pth')
    # model1.load_state_dict(model1_pth, strict=True)
    model_cd2.load_state_dict(model2_pth, strict=True)
    # 语义分割模型
    model_seg = FarSeg(backbone="resnet50", classes=2,
                       backbone_pretrained=False)
    model_seg_pth = torch.load('test_model/farseg_resnet50_seg_8.03.pth')

    model_seg.load_state_dict(model_seg_pth, strict=True)

    models = [model_cd2, model_seg]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    cmap = color_map()
    tbar = tqdm(testloader)
    # if not os.path.exists(sys.argv[2]):
    #     os.mkdir(sys.argv[2])
    if not os.path.exists('./output_path'):
        os.mkdir('./output_path')
    from utils.metric import F1_score
    metric_bin = F1_score(num_classes=2)
    # 开始测试
    with torch.no_grad():
        for k, (img1, img2, mask, id) in enumerate(tbar):
            img1, img2 = img1.cuda(
                non_blocking=True), img2.cuda(non_blocking=True)

            # out1_list, out2_list, out_bin_list = [], [], []
            # for model in models:
            #     out1, out2, out_bin = model(img1, img2, True)
            #     out1_list.append(out1)
            #     out2_list.append(out2)
            #     out_bin_list.append(out_bin)
            # out1 = model_seg(img1, True)
            # out2 = model_seg(img2, True)

            out_list = []
            # for model in models:
            # out_cd1 = model1(img1, img2)
            # out_cd1 = torch.unsqueeze(out_cd1[:, 1, :, :], dim=1)
            img = torch.cat([torch.unsqueeze(img1, dim=1),
                             torch.unsqueeze(img2, dim=1)], dim=1)
            # out_cd1 = model1(img)['change_prob']
            out_cd2 = model_cd2(img)['change_prob']
            # out_list.append(out_cd1)
            # out_list.append(out_cd2)
            # # 将不同模型的预测结果合并
            # out = torch.stack(out_list, dim=0)
            # # 计算不同模型预测结果的均值
            # out = torch.sum(out, dim=0) / 1
            # (bz, 512, 512)
            # out1 = torch.argmax(out1, dim=1).cpu().numpy()
            # out2 = torch.argmax(out2, dim=1).cpu().numpy()
            # out_bin = torch.argmax(out_bin, dim=1).cpu().numpy()
            out_bin = (out_cd2 > 0.5).cpu().numpy().astype(np.uint8)
            metric_bin.add_batch(out_bin, mask.numpy())
            # 保存预测结果
            # for i in range(out1.shape[0]):
            #     mask1 = Image.fromarray(out1[i].astype(np.uint8), mode="P")
            #     mask1.putpalette(cmap)
            #     mask1.save(sys.argv[2] + "/" +
            #                id[i]+"_1_label.png")
            #     mask2 = Image.fromarray(out2[i].astype(np.uint8), mode="P")
            #     mask2.putpalette(cmap)
            #     mask2.save(sys.argv[2] + "/" +
            #                id[i]+"_2_label.png")
            #     mask_bin = Image.fromarray(
            #         out_bin[i].astype(np.uint8), mode="P")
            #     mask_bin.putpalette(cmap)
            #     mask_bin.save(sys.argv[2] + "/" +
            #                   id[i]+"_change.png")
        f1_bin = metric_bin.evaluate()
        Score = 0.6*f1_bin
        print(
            "f1_bin:{}  Score:{}".format(f1_bin, Score))
    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))
