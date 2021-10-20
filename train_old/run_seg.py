from datasets.segmentation import Segmentation
from models.model_zoo import get_model
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
from torchgeo1.models import FarSeg
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import multi_scale
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
    # 标识
    MODE = "val"
    # 获取参数
    # with open("exp_result/unet_resnet34_notta_2021-09-17-15:16/logs/model_cfg.json") as fp:
    #     args1 = json.load(fp)
    # with open("val_model/fcn_resnet34_notta_focal_2021-09-22-10:12/logs/model_cfg.json") as fp:
    #     args2 = json.load(fp)
    # print(args)
    # args = Options().parse()
    torch.backends.cudnn.benchmark = True

    # args1 = dict_to_object(args1)
    # args2 = dict_to_object(args2)
    # print(args.model, args.backbone, args.pretrained, args.lightweight)
    print(torch.cuda.is_available())
    # 加载数据
    valset = Segmentation("/home/kelin/data/train", mode="val")

    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           pin_memory=True, num_workers=4, drop_last=False)
    # 加载模型
    # model1 = get_model(args1.model, args1.backbone, False,
    #                    len(valset.CLASSES)-1, args1.lightweight)
    model1 = FarSeg(backbone="resnet50", classes=2, backbone_pretrained=False)
    # model1 = DataParallel(model1)
    # model2 = get_model(args2.model, args2.backbone, False,
    #                    len(valset.CLASSES)-1, args2.lightweight)
    # print(model1)
    # 加载权重文件
    model_change_pth = torch.load(
        'exp_result/farseg_resnet50_seg_v2_2021-10-02-14:20/checkpoints/farseg_resnet50_seg_8.03.pth')
    model1.load_state_dict(model_change_pth, strict=True)

    # model2_pth = torch.load(
    #     'val_model/fcn_resnet34_notta_focal_2021-09-22-10:12/checkpoints/fcn_resnet34_68.43.pth')
    # model2.load_state_dict(model2_pth, strict=True)

    # models = [model1, model2]
    models = [model1]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    cmap = color_map()

    tbar = tqdm(valloader)

    from utils.metric import F1_score
    from utils.image import visualize
    if not os.path.exists('./output_path'):
        os.mkdir('./output_path')
    metric_seg = F1_score(num_classes=2)
    with torch.no_grad():
        for i, (img, label, id) in enumerate(tbar):
            img = img.cuda(non_blocking=True)
            mask_seg = label.cuda(non_blocking=True)
            out1_list, out2_list, out_bin_list = [], [], []
            for model in models:
                out = multi_scale.multi_scale_inference(model, img)
                # out = model(img)
                # print(out1.shape)
                # sys.exit()
                # out1_list.append(out1)
                # out2_list.append(out2)
                # out_bin_list.append(out_bin)
            # 将不同模型的预测结果合并
            # out1 = torch.stack(out1_list, dim=0)
            # # 计算不同模型预测结果的均值
            # out1 = torch.sum(out1, dim=0) / len(models)
            # # 将不同模型的预测结果合并
            # out2 = torch.stack(out2_list, dim=0)
            # # 计算不同模型预测结果的均值
            # out2 = torch.sum(out2, dim=0) / len(models)
            # # 将不同模型的预测结果合并
            # out_bin = torch.stack(out_bin_list, dim=0)
            # # 计算不同模型预测结果的均值
            # out_bin = torch.sum(out_bin, dim=0) / len(models)
            # print(out1.shape) torch.Size([1, 2, 512, 512])
            # (bz,h,w)
            out = torch.argmax(out, dim=1).cpu().numpy()
            # 计算预测精度
            # metric1.add_batch(out1, mask1.cpu().numpy())
            # metric2.add_batch(out2, mask2.cpu().numpy())
            metric_seg.add_batch(out, label.cpu().numpy())
            # if k % 20 == 0:
            #     visualize([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
            #               [mask1[0].cpu().numpy(),
            #                mask2[0].cpu().numpy()],
            #               [out1[0], out2[0]],
            #               [mask_bin[0].cpu().numpy(), out_bin[0]],
            #               "./output_path", id[0])
            # 保存预测结果
            # for i in range(out1.shape[0]):
            #     mask1 = Image.fromarray(out1[i].astype(np.uint8), mode="P")
            #     mask1.putpalette(cmap)
            #     # mask1.save("./output_path" + "/" +
            #     #            id[i].split(".")[0]+"_label1.png")

            #     mask2 = Image.fromarray(out2[i].astype(np.uint8), mode="P")
            #     mask2.putpalette(cmap)
            #     mask2.save("./output_path" + "/" +
            #                id[i].split(".")[0]+"_label2.png")

            #     mask_bin = Image.fromarray(
            #         out_bin[i].astype(np.uint8), mode="P")
            #     mask_bin.putpalette(cmap)
            # mask_bin.save("./output_path" + "/" +
            #               id[i].split(".")[0].split("_")[0]+"_change.png")

        # 计算F1-Score
        f1_seg = metric_seg.evaluate()
        Score = 0.1 * f1_seg
        print(
            "f1_seg:{}  Score:{}".format(f1_seg, Score))
    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))
