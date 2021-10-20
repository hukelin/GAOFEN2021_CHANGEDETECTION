from models.backbone.hrnet import HRNet
from models.backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d

import torch
from torch import nn
import torch.nn.functional as F
from models.pointrend import PointHead

from models.block.attention import PAM_Module, CAM_Module
from efficientnet_pytorch import EfficientNet


def get_backbone(backbone, pretrained):
    if backbone == "resnet18":
        backbone = resnet18(pretrained)
    elif backbone == "resnet34":
        backbone = resnet34(pretrained)
    elif backbone == "resnet50":
        backbone = resnet50(pretrained)
    elif backbone == "resnet101":
        backbone = resnet101(pretrained)
    elif backbone == "resnet152":
        backbone = resnet152(pretrained)
    elif backbone == "resnext50":
        backbone = resnext50_32x4d(pretrained)
    elif backbone == "resnext101":
        backbone = resnext101_32x8d(pretrained)
    elif "hrnet" in backbone:
        backbone = HRNet(backbone, pretrained)
    elif "efficientnet-b3":
        backbone = EfficientNet.from_pretrained('efficientnet-b3')
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone


# class BaseNet(nn.Module):
#     def __init__(self, backbone, pretrained):
#         super(BaseNet, self).__init__()
#         self.backbone = get_backbone(backbone, pretrained)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1 = self.head(x1)
#         out2 = self.head(x2)

#         out1 = F.interpolate(out1, size=(
#             h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(
#             h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = F.interpolate(out_bin, size=(
#             h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.softmax(out_bin)

#         return out1, out2, out_bin.squeeze(1)

#     def forward(self, x1, x2, tta=False):
#         if not tta:
#             return self.base_forward(x1, x2)
#         else:
#             out1, out2, out_bin = self.base_forward(x1, x2)
#             out1 = F.softmax(out1, dim=1)
#             out2 = F.softmax(out2, dim=1)
#             out_bin = out_bin.unsqueeze(1)
#             origin_x1 = x1.clone()
#             origin_x2 = x2.clone()

#             x1 = origin_x1.flip(2)
#             x2 = origin_x2.flip(2)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += F.softmax(cur_out1, dim=1).flip(2)
#             out2 += F.softmax(cur_out2, dim=1).flip(2)
#             out_bin += cur_out_bin.unsqueeze(1).flip(2)

#             x1 = origin_x1.flip(3)
#             x2 = origin_x2.flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += F.softmax(cur_out1, dim=1).flip(3)
#             out2 += F.softmax(cur_out2, dim=1).flip(3)
#             out_bin += cur_out_bin.unsqueeze(1).flip(3)

#             x1 = origin_x1.transpose(2, 3).flip(3)
#             x2 = origin_x2.transpose(2, 3).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)
#             out2 += F.softmax(cur_out2, dim=1).flip(3).transpose(2, 3)
#             out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

#             x1 = origin_x1.flip(3).transpose(2, 3)
#             x2 = origin_x2.flip(3).transpose(2, 3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)
#             out2 += F.softmax(cur_out2, dim=1).transpose(2, 3).flip(3)
#             out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

#             x1 = origin_x1.flip(2).flip(3)
#             x2 = origin_x2.flip(2).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)
#             out2 += F.softmax(cur_out2, dim=1).flip(3).flip(2)
#             out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

#             out1 /= 6.0
#             out2 /= 6.0
#             out_bin /= 6.0

#             return out1, out2, out_bin.squeeze(1)
# class BaseNet(nn.Module):
#     def __init__(self, backbone, pretrained):
#         super(BaseNet, self).__init__()
#         self.backbone = get_backbone(backbone, pretrained)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape
#         # TODO 改动，直接输入二者的差值
#         x_bin = x2-x1
#         # backbone提取特征
#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]
#         # head输出
#         out1 = self.head(x1)
#         out2 = self.head(x2)
#         # 上采样至原图像大小
#         out1 = F.interpolate(out1, size=(
#             h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(
#             h, w), mode='bilinear', align_corners=False)

#         # softmax输出
#         out1 = torch.softmax(out1)
#         out2 = torch.softmax(out2)

#         # 输出change，并上采样
#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = F.interpolate(out_bin, size=(
#             h, w), mode='bilinear', align_corners=False)

#         # softmax输出
#         out_bin = torch.softmax(out_bin)

#         return out1, out2, out_bin

#     def forward(self, x1, x2, tta=False):
#         # 不加TTA
#         if not tta:
#             # 调用子类的base_forward方法
#             return self.base_forward(x1, x2)
#         # 加TTA
#         else:
#             # 原图像输出
#             out1, out2, out_bin = self.base_forward(x1, x2)
#             # 原图像
#             origin_x1 = x1.clone()
#             origin_x2 = x2.clone()

#             # 对dim=2翻转后输出
#             x1 = origin_x1.flip(2)
#             x2 = origin_x2.flip(2)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             # 叠加输出
#             out1 += cur_out1.flip(2)
#             out2 += cur_out2.flip(2)
#             out_bin += cur_out_bin.flip(2)

#             # 对dim=3翻转后输出
#             x1 = origin_x1.flip(3)
#             x2 = origin_x2.flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += cur_out1.flip(3)
#             out2 += cur_out2.flip(3)
#             out_bin += cur_out_bin.flip(3)

#             # 换轴再翻转
#             x1 = origin_x1.transpose(2, 3).flip(3)
#             x2 = origin_x2.transpose(2, 3).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += cur_out1.flip(3).transpose(2, 3)
#             out2 += cur_out2.flip(3).transpose(2, 3)
#             out_bin += cur_out_bin.flip(3).transpose(2, 3)

#             # 翻转再换轴
#             x1 = origin_x1.flip(3).transpose(2, 3)
#             x2 = origin_x2.flip(3).transpose(2, 3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += cur_out1.transpose(2, 3).flip(3)
#             out2 += cur_out2.transpose(2, 3).flip(3)
#             out_bin += cur_out_bin.transpose(2, 3).flip(3)

#             # 同时翻转dim=2和dim=3
#             x1 = origin_x1.flip(2).flip(3)
#             x2 = origin_x2.flip(2).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
#             out1 += cur_out1.flip(3).flip(2)
#             out2 += cur_out2.flip(3).flip(2)
#             out_bin += cur_out_bin.flip(3).flip(2)

#             # 计算TTA输出均值
#             out1 /= 6.0
#             out2 /= 6.0
#             out_bin /= 6.0

#             return out1, out2, out_bin


# ***************************backone==resnet****************************
class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape
        # backbone提取双时相特征
        features1 = self.backbone.base_forward(x1)
        features2 = self.backbone.base_forward(x2)
        # 输出change，并上采样
        out_bin = torch.abs(features2[-1] - features1[-1])
        out_bin = self.head(out_bin)
        out_bin = F.interpolate(out_bin, size=(
            h, w), mode='bilinear', align_corners=True)

        # out_bin = torch.softmax(out_bin, dim=1)

        return out_bin

    def forward(self, x1, x2, tta=False):
        # 不加TTA
        if not tta:
            # 调用子类的base_forward方法
            return self.base_forward(x1, x2)
        # 加TTA
        else:
            # 原图像输出
            out1, out2, out_bin = self.base_forward(
                x1, x2)
            # 原图像
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            # 对dim=2翻转后输出
            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(
                x1, x2)
            # 叠加输出
            out1 += cur_out1.flip(2)
            out2 += cur_out2.flip(2)
            out_bin += cur_out_bin.flip(2)

            # 对dim=3翻转后输出
            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(
                x1, x2)
            out1 += cur_out1.flip(3)
            out2 += cur_out2.flip(3)
            out_bin += cur_out_bin.flip(3)

            # 换轴再翻转
            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(
                x1, x2)
            out1 += cur_out1.flip(3).transpose(2, 3)
            out2 += cur_out2.flip(3).transpose(2, 3)
            out_bin += cur_out_bin.flip(3).transpose(2, 3)

            # 翻转再换轴
            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(
                x1, x2)
            out1 += cur_out1.transpose(2, 3).flip(3)
            out2 += cur_out2.transpose(2, 3).flip(3)
            out_bin += cur_out_bin.transpose(2, 3).flip(3)

            # 同时翻转dim=2和dim=3
            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(
                x1, x2)
            out1 += cur_out1.flip(3).flip(2)
            out2 += cur_out2.flip(3).flip(2)
            out_bin += cur_out_bin.flip(3).flip(2)

            # 计算TTA输出均值
            out1 /= 6.0
            out2 /= 6.0
            out_bin /= 6.0

            return out1, out2, out_bin
# ***************************backone==resnet***************************


# class BaseNet(nn.Module):
#     def __init__(self, backbone, pretrained):
#         super(BaseNet, self).__init__()
#         # self.sa = PAM_Module(1536).cuda()
#         # self.sc = CAM_Module(1536).cuda()
#         self.backbone = get_backbone(backbone, pretrained)
#         self.point_head = PointHead(in_c=1538)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape
#         # backbone提取特征
#         features1 = self.backbone.base_forward(x1)
#         features2 = self.backbone.base_forward(x2)
#         # sa1 = self.sa(features1)
#         # sc1 = self.sc(features1)
#         # sa2 = self.sa(features2)
#         # sc2 = self.sc(features2)
#         # features1 = sa1 + sc1
#         # features2 = sa2 + sc2

#         # head输出
#         out1 = self.head(features1)
#         out2 = self.head(features2)
#         out_point1 = self.point_head(x1, features1, out1)
#         out_point2 = self.point_head(x2, features2, out2)

#         # 上采样至原图像大小
#         out1 = F.interpolate(out1, size=(
#             h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(
#             h, w), mode='bilinear', align_corners=False)

#         # softmax输出
#         out1 = torch.softmax(out1, dim=1)
#         out2 = torch.softmax(out2, dim=1)

#         # 输出change，并上采样
#         out_bin = torch.abs(features2 - features1)
#         out_bin = self.head_bin(out_bin)
#         out_bin = F.interpolate(out_bin, size=(
#             h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.softmax(out_bin, dim=1)

#         return out1, out2, out_bin

#     def forward(self, x1, x2, tta=False):
#         # 不加TTA
#         if not tta:
#             # 调用子类的base_forward方法
#             return self.base_forward(x1, x2)
#         # 加TTA
#         else:
#             # 原图像输出
#             out1, out2, out_bin = self.base_forward(
#                 x1, x2)
#             # 原图像
#             origin_x1 = x1.clone()
#             origin_x2 = x2.clone()

#             # 对dim=2翻转后输出
#             x1 = origin_x1.flip(2)
#             x2 = origin_x2.flip(2)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(
#                 x1, x2)
#             # 叠加输出
#             out1 += cur_out1.flip(2)
#             out2 += cur_out2.flip(2)
#             out_bin += cur_out_bin.flip(2)

#             # 对dim=3翻转后输出
#             x1 = origin_x1.flip(3)
#             x2 = origin_x2.flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(
#                 x1, x2)
#             out1 += cur_out1.flip(3)
#             out2 += cur_out2.flip(3)
#             out_bin += cur_out_bin.flip(3)

#             # 换轴再翻转
#             x1 = origin_x1.transpose(2, 3).flip(3)
#             x2 = origin_x2.transpose(2, 3).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(
#                 x1, x2)
#             out1 += cur_out1.flip(3).transpose(2, 3)
#             out2 += cur_out2.flip(3).transpose(2, 3)
#             out_bin += cur_out_bin.flip(3).transpose(2, 3)

#             # 翻转再换轴
#             x1 = origin_x1.flip(3).transpose(2, 3)
#             x2 = origin_x2.flip(3).transpose(2, 3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(
#                 x1, x2)
#             out1 += cur_out1.transpose(2, 3).flip(3)
#             out2 += cur_out2.transpose(2, 3).flip(3)
#             out_bin += cur_out_bin.transpose(2, 3).flip(3)

#             # 同时翻转dim=2和dim=3
#             x1 = origin_x1.flip(2).flip(3)
#             x2 = origin_x2.flip(2).flip(3)
#             cur_out1, cur_out2, cur_out_bin = self.base_forward(
#                 x1, x2)
#             out1 += cur_out1.flip(3).flip(2)
#             out2 += cur_out2.flip(3).flip(2)
#             out_bin += cur_out_bin.flip(3).flip(2)

#             # 计算TTA输出均值
#             out1 /= 6.0
#             out2 /= 6.0
#             out_bin /= 6.0

#             return out1, out2, out_bin
