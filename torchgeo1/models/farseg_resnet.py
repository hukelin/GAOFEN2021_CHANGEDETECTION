# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Foreground-Aware Relation Network (FarSeg) implementations."""
import math
from collections import OrderedDict
from typing import List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d,
)
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN
import torch.hub
import torch.hub
import torch
import sys
# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"
ModuleList.__module__ = "nn.ModuleList"
Sequential.__module__ = "nn.Sequential"
Conv2d.__module__ = "nn.Conv2d"
BatchNorm2d.__module__ = "nn.BatchNorm2d"
ReLU.__module__ = "nn.ReLU"
UpsamplingBilinear2d.__module__ = "nn.UpsamplingBilinear2d"
Sigmoid.__module__ = "nn.Sigmoid"
Identity.__module__ = "nn.Identity"


class FarSeg(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature presentation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        classes: int = 16,
        backbone_pretrained: bool = True
    ) -> None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        super(FarSeg, self).__init__()  # type: ignore[no-untyped-call]

        if backbone in ["resnet18", "resnet34"]:
            max_channels = 512
        elif backbone in ["resnet50", "resnet101"]:
            max_channels = 2048
        elif backbone in ["resnext50_32x4d", "resnext101_32x8d"]:
            max_channels = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}.")
        self.backbone = getattr(resnet, backbone)(
            pretrained=backbone_pretrained)
        self.fpn = FPN(
            in_channels_list=[max_channels //
                              (2 ** (3 - i)) for i in range(4)],  # [256, 512, 1024, 2048]
            out_channels=256,
        )
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)

    def extract_features(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        torch.Size([2, 256, 128, 128])
        torch.Size([2, 512, 64, 64])
        torch.Size([2, 1024, 32, 32])
        torch.Size([2, 2048, 16, 16])
        features = [c2, c3, c4, c5]
        # for i in features:
        #     print(i.shape)
        # features = self.backbone(x)
        # for i, j in enumerate(features):
        #     print(i, j.shape)
        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(OrderedDict(
            {f"c{i + 2}": features[i] for i in range(len(features))}))
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)

        logit = self.decoder(features)

        return cast(Tensor, logit)

    def base_forward(self, x: Tensor) -> Tensor:
        out = self.extract_features(x)
        out = torch.softmax(out, dim=1)
        return out

    def forward(self, x: Tensor, tta=False) -> Tensor:
        if not tta:
            out = self.base_forward(x)
        else:
            # 原图像输出
            out = self.base_forward(x)
            # 原图像
            origin_x = x.clone()
            # 对dim=2翻转后输出
            x = origin_x.flip(2)
            cur_out = self.base_forward(x)
            # 叠加输出
            out += cur_out.flip(2)
            # 对dim=3翻转后输出
            x = origin_x.flip(3)
            cur_out = self.base_forward(x)
            out += cur_out.flip(3)

            # 换轴再翻转
            x = origin_x.transpose(2, 3).flip(3)
            cur_out = self.base_forward(x)
            out += cur_out.flip(3).transpose(2, 3)

            # 翻转再换轴
            x = origin_x.flip(3).transpose(2, 3)
            cur_out = self.base_forward(x)
            out += cur_out.transpose(2, 3).flip(3)

            # 同时翻转dim=2和dim=3
            cur_out = self.base_forward(x)
            out += cur_out.flip(3).flip(2)

            # 计算TTA输出均值
            out /= 6.0
            out = torch.softmax(out, dim=1)
        return out
# **********************添加边界约束**********************
# class FarSeg(Module):
#     """Foreground-Aware Relation Network (FarSeg).

#     This model can be used for binary- or multi-class object segmentation, such as
#     building, road, ship, and airplane segmentation. It can be also extended as a change
#     detection model. It features a foreground-scene relation module to model the
#     relation between scene embedding, object context, and object feature, thus improving
#     the discrimination of object feature presentation.

#     If you use this model in your research, please cite the following paper:

#     * https://arxiv.org/pdf/2011.09766.pdf
#     """

#     def __init__(
#         self,
#         backbone: str = "resnet50",
#         classes: int = 16,
#         backbone_pretrained: bool = True,
#     ) -> None:
#         """Initialize a new FarSeg model.

#         Args:
#             backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
#                 "resnet50", "resnet101"]
#             classes: number of output segmentation classes
#             backbone_pretrained: whether to use pretrained weight for backbone
#         """
#         super().__init__()  # type: ignore[no-untyped-call]
#         if backbone in ["resnet18", "resnet34"]:
#             max_channels = 512
#         elif backbone in ["resnet50", "resnet101"]:
#             max_channels = 2048
#         else:
#             raise ValueError(f"unknown backbone: {backbone}.")
#         self.backbone = getattr(resnet, backbone)(
#             pretrained=backbone_pretrained)

#         self.fpn = FPN(
#             in_channels_list=[max_channels //
#                               (2 ** (3 - i)) for i in range(4)],
#             out_channels=256,
#         )
#         self.fsr = _FSRelation(max_channels, [256] * 4, 256)
#         self.decoder = _LightWeightDecoder(256, 128, classes)
#         self.conv_2block = nn.Sequential(
#             nn.Conv2d(2, 32, 1, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 1, stride=1),
#             nn.BatchNorm2d(32)
#         )
#         self.conv_final = nn.Conv2d(32, 2, 1, stride=1)

#     def base_forward(self, x: Tensor) -> Tensor:
#         """Forward pass of the model.

#         Args:
#             x: input image

#         Returns:
#             output prediction
#         """
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         c2 = self.backbone.layer1(x)
#         c3 = self.backbone.layer2(c2)
#         c4 = self.backbone.layer3(c3)
#         c5 = self.backbone.layer4(c4)
#         features = [c2, c3, c4, c5]

#         coarsest_features = features[-1]
#         scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
#         fpn_features = self.fpn(
#             OrderedDict({f"c{i + 2}": features[i] for i in range(4)})
#         )
#         features = [v for k, v in fpn_features.items()]
#         features = self.fsr(scene_embedding, features)
#         logit = self.decoder(features)

#         return features, cast(Tensor, logit)

#     def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
#         features1, out1 = self.base_forward(x1)
#         features2, out2 = self.base_forward(x2)
#         features_bin = [torch.abs(j-i) for i, j in zip(features1, features2)]
#         out_bin = self.decoder(features_bin)

#         out1 = torch.softmax(out1, dim=1)
#         out2 = torch.softmax(out2, dim=1)
#         out_bin = torch.softmax(out_bin, dim=1)

#         out_e1 = self.conv_2block(out1)
#         out_e1 = self.conv_final(out_e1)

#         out_e2 = self.conv_2block(out2)
#         out_e2 = self.conv_final(out_e2)

#         out_ec = self.conv_2block(out_bin)
#         out_ec = self.conv_final(out_ec)

#         out_e1 = torch.softmax(out_e1, dim=1)
#         out_e2 = torch.softmax(out_e2, dim=1)
#         out_ec = torch.softmax(out_ec, dim=1)

#         out1 = 1+out1-out_e1
#         out2 = 1+out2-out_e2
#         out_bin = 1+out_bin-out_ec

#         out1 = torch.softmax(out1, dim=1)
#         out2 = torch.softmax(out2, dim=1)
#         out_bin = torch.softmax(out_bin, dim=1)

#         return out1, out2, out_bin, out_e1, out_e2, out_ec


# *****************************变化检测*****************************
class FarSeg_CD(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature presentation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        classes: int = 16,
        backbone_pretrained: bool = True
    ) -> None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        super(FarSeg_CD, self).__init__()  # type: ignore[no-untyped-call]
        if backbone in ["resnet18", "resnet34"]:
            max_channels = 512
        elif backbone in ["resnet50", "resnet101"]:
            max_channels = 2048
        elif backbone in ["resnext50_32x4d", "resnext101_32x8d"]:
            max_channels = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}.")
        self.backbone = getattr(resnet, backbone)(
            pretrained=backbone_pretrained)

        self.fpn = FPN(
            in_channels_list=[max_channels //
                              (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)

    def extract_features(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)  # torch.Size([2, 64, 256, 256])
        x = self.backbone.maxpool(x)  # torch.Size([2, 64, 128, 128])
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]
        # features = self.backbone(x)
        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(
            OrderedDict({f"c{i + 2}": features[i] for i in range(4)})
        )
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)
        logit = self.decoder(features)

        return features, cast(Tensor, logit)

    def base_forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        features1, out1 = self.extract_features(x1)
        features2, out2 = self.extract_features(x2)
        features_bin = [torch.abs(j-i) for i, j in zip(features1, features2)]
        out_bin = self.decoder(features_bin)
        out = torch.softmax(out_bin, dim=1)
        return out

    def forward(self, x1: Tensor, x2: Tensor, tta=False) -> Tensor:
        if not tta:
            out_bin = self.base_forward(x1, x2)
        else:
            # 原图像输出
            out_bin = self.base_forward(
                x1, x2)
            # 原图像
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            # 对dim=2翻转后输出
            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out_bin = self.base_forward(
                x1, x2)
            # 叠加输出
            out_bin += cur_out_bin.flip(2)

            # 对dim=3翻转后输出
            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out_bin = self.base_forward(
                x1, x2)
            out_bin += cur_out_bin.flip(3)

            # 换轴再翻转
            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out_bin = self.base_forward(
                x1, x2)
            out_bin += cur_out_bin.flip(3).transpose(2, 3)

            # 翻转再换轴
            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out_bin = self.base_forward(x1, x2)

            out_bin += cur_out_bin.transpose(2, 3).flip(3)

            # 同时翻转dim=2和dim=3
            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out_bin = self.base_forward(x1, x2)
            out_bin += cur_out_bin.flip(3).flip(2)
            # 计算TTA输出均值
            out_bin /= 6.0
            out_bin = torch.softmax(out_bin, dim=1)
        return out_bin


class _FSRelation(Module):
    """F-S Relation module."""

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        """Initialize the _FSRelation module.

        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.scene_encoder = ModuleList(
            [
                Sequential(
                    Conv2d(scene_embedding_channels, out_channels, 1),
                    ReLU(True),
                    Conv2d(out_channels, out_channels, 1),
                )
                for _ in range(len(in_channels_list))
            ]
        )

        self.content_encoders = ModuleList()
        self.feature_reencoders = ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1),
                    BatchNorm2d(out_channels),  # type: ignore[no-untyped-call]
                    ReLU(True),
                )
            )
            self.feature_reencoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1),
                    BatchNorm2d(out_channels),  # type: ignore[no-untyped-call]
                    ReLU(True),
                )
            )

        self.normalizer = Sigmoid()  # type: ignore[no-untyped-call]

    def forward(self, scene_feature: Tensor, features: List[Tensor]) -> List[Tensor]:
        """Forward pass of the model."""
        # [N, C, H, W]
        content_feats = [
            c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)
        ]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]

        p_feats = [op(p_feat)
                   for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class _LightWeightDecoder(Module):
    """Light Weight Decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        in_feature_output_strides: List[int] = [4, 8, 16, 32],
        out_feature_output_stride: int = 4,
    ) -> None:
        """Initialize the _LightWeightDecoder module.

        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different
                levels
            out_feature_output_stride: output stride of output feature maps
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.blocks = ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feature_output_stride))
            )
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                Sequential(
                    *[
                        Sequential(
                            Conv2d(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            # type: ignore[no-untyped-call]
                            BatchNorm2d(out_channels),
                            ReLU(inplace=True),
                            UpsamplingBilinear2d(scale_factor=2)
                            if num_upsample != 0
                            else Identity(),  # type: ignore[no-untyped-call]
                        )
                        for idx in range(num_layers)
                    ]
                )
            )

        self.classifier = Sequential(
            Conv2d(out_channels, num_classes, 3, 1, 1),
            UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward pass of the model."""
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out_feat = self.classifier(out_feat)

        return cast(Tensor, out_feat)


class _LightWeightDecoder2(Module):
    """Light Weight Decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        in_feature_output_strides: List[int] = [4, 8, 16, 32],
        out_feature_output_stride: int = 4,
    ) -> None:
        """Initialize the _LightWeightDecoder module.
        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different
                levels
            out_feature_output_stride: output stride of output feature maps
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.blocks = ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - \
                int(math.log2(int(out_feature_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                Sequential(
                    *[
                        Sequential(
                            Conv2d(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            # type: ignore[no-untyped-call]
                            BatchNorm2d(out_channels),
                            ReLU(inplace=True),
                            UpsamplingBilinear2d(scale_factor=2)
                            if num_upsample != 0
                            else Identity(),  # type: ignore[no-untyped-call]
                        )
                        for idx in range(num_layers)
                    ]
                )
            )

        self.classifier = Sequential(
            Conv2d(out_channels, num_classes, 3, 1, 1),
            UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward pass of the model."""
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)
        # for i, j in enumerate(inner_feat_list):
        #     print(i, j.shape)
        # 0 torch.Size([2, 128, 128, 128])
        # 1 torch.Size([2, 128, 128, 128])
        # 2 torch.Size([2, 128, 128, 128])
        # 3 torch.Size([2, 128, 128, 128])
        # torch.Size([2, 128, 128, 128])
        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out = []
        for i in inner_feat_list:
            out.append(self.classifier(i))
        # out_feat = torch.cat(inner_feat_list, dim=1)
        out_feat = self.classifier(out_feat)
        out.append(out_feat)
        # return cast(Tensor, out_feat)
        return out
