
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from skimage.morphology import binary_dilation, disk
# import torchcontrib
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from loss import OhemCrossEntropy
from datasets.change_detection import ChangeDetection
from loss import annealing_softmax_focalloss
from models.model_zoo import get_model
from models.pointrend import point_sample
from torchgeo1.models import FarSeg_CD, FarSeg_CD_Res
from torchgeo.models import ChangeStarFarSeg, FCEF, FCSiamConc
from utils.image import add_image, visualize, visualize_CD, visualize_edge
from utils.metric import F1_score
from utils.options import Options
from utils.palette import color_map

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("\n")


def batch_ce_loss(y_pred, y_true, l_weight=1.0):
    label = y_true.detach().cpu().numpy()
    # 统计类别数量
    label_num = np.bincount(label.flatten())
    weights = label_num.sum() / (label_num * 2)
    if len(weights) == 1:
        weights = np.append(weights, 0.)
    weights = torch.FloatTensor(weights).cuda()
    loss = CrossEntropyLoss(weight=weights, reduction="mean")(y_pred, y_true)
    return l_weight*loss


# 加载训练参数
args = Options().parse()
# 设置实验保存文件夹
time_str = time.strftime('%Y-%m-%d-%H:%M')
args.exp_name = '{}_{}_{}_{}'.format(
    args.model, args.backbone, args.exp_name, time_str)
# 日志记录路径
logs_dir = os.path.join('exp_result_original_data', args.exp_name, 'logs')
# 日志记录文件
logger.add("{}/val_log.log".format(logs_dir))


class Trainer:
    def __init__(self, args):
        self.args = args
        # 加载数据
        trainset = ChangeDetection(root=args.train_root, mode="train")
        valset = ChangeDetection(root=args.val_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=True,
                                    pin_memory=True, num_workers=4, drop_last=False)
        logger.info("train samples:{}".format(len(trainset)))
        logger.info("val samples:{}".format(len(valset)))
        # *******************************模型选择*************************************
        # self.model = get_model(args.model, args.backbone,
        #                        args.pretrained, 2, True)
        # if args.backbone == "hrnet_w30":
        #     self.model = FarSeg_CD(
        #         backbone=args.backbone, classes=2, backbone_pretrained=False)
        # elif args.backbone == "resnext50_32x4d":
        #     self.model = FarSeg_CD_Res(backbone=args.backbone, classes=2)
        self.model = FCEF()
        # *******************************模型选择*************************************
        # 加载预训练模型
        if args.pretrain_from:
            print("hello")
            self.model.load_state_dict(
                torch.load(args.pretrain_from), strict=True)
        self.model = DataParallel(self.model)
        # *****************************损失函数的选择********************************
        # 二分类
        weights = torch.FloatTensor([1, 2]).cuda()
        self.criterion_bin = CrossEntropyLoss(weight=weights)
        # self.criterion_bin = batch_ce_loss
        # self.criterion = annealing_softmax_focalloss
        # self.criterion = DiceLoss("multiclass")
        # *****************************损失函数的选择********************************
        #
        # param = [name for name, param in self.model.named_parameters()]
        # print(param)
        # sys.exit()
        # 设置优化器
        if args.optimizer == "adamw":
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(),
                                 lr=args.lr,
                                 momentum=0.9,
                                 weight_decay=args.weight_decay,
                                 nesterov=False
                                 )
        # 调整学习率
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=3,  # T_0就是初始restart的epoch数目
            T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 + T_0 * T_mult
            eta_min=1e-4  # 最低学习率
        )
        # 模型并行训练
        # self.model = DataParallel(self.model).cuda()
        self.model = self.model.cuda()
        # print(next(self.model.parameters()).device)
        self.iters = 0
        # 迭代总次数
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        loss_bin2 = 0.0
        for i, (img1, img2, mask_bin, id) in enumerate(tbar):
            img1, img2 = img1.cuda(), img2.cuda()
            mask_bin = mask_bin.cuda()
            img = torch.cat([torch.unsqueeze(img1, dim=1),
                             torch.unsqueeze(img2, dim=1)], dim=1)
            # print(img1.shape)  # torch.Size([4, 3, 512, 512])
            # print(img2.shape)  # torch.Size([4, 3, 512, 512])
            # print(mask_bin.shape)  # torch.Size([4, 512, 512])
            # 模型输出
            out_bin = self.model(img)
            loss_bin1 = self.criterion_bin(out_bin.squeeze(), mask_bin.long())
            # loss_bin2 = self.criterion(
            #     out_bin.squeeze(), mask_bin.long(), self.iters, self.total_iters)
            # loss_bin2 = self.criterion(out_bin.squeeze(), mask_bin.long())
            loss = loss_bin1+loss_bin2
            loss = loss.mean()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iters += 1

            # lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            # self.optimizer.param_groups[0]["lr"] = lr
            # self.optimizer.param_groups[1]["lr"] = lr * 10.0
            tbar.set_description("Train Loss: %.3f" % (total_loss / (i + 1)))

        train_writer.add_scalar("loss", total_loss, epoch)
        logger.info("Train Loss:{}".format(total_loss / (i + 1)))

    def validation(self):
        total_loss = 0.0
        tbar = tqdm(self.valloader)
        self.model.eval()
        loss_bin2 = 0.0
        metric_change = F1_score(num_classes=2)
        with torch.no_grad():
            for i, (img1, img2,  mask_bin, id) in enumerate(tbar):
                img1, img2 = img1.cuda(), img2.cuda()
                mask_bin = mask_bin.cuda()
                img = torch.cat([torch.unsqueeze(img1, dim=1),
                                 torch.unsqueeze(img2, dim=1)], dim=1)
                # print(img1.shape)  # torch.Size([4, 3, 512, 512])
                # print(img2.shape)  # torch.Size([4, 3, 512, 512])
                # print(mask_bin.shape)  # torch.Size([4, 512, 512])
                # 模型输出
                out_bin = self.model(img)
                loss_bin1 = self.criterion_bin(
                    out_bin.squeeze(), mask_bin.long())
                # loss_bin2 = self.criterion(
                #     out_bin.squeeze(), mask_bin.long(), self.iters, self.total_iters)
                # loss_bin2 = self.criterion(out_bin.squeeze(), mask_bin.long())
                loss = loss_bin1+loss_bin2
                loss = loss.mean()
                total_loss += loss.item()
                tbar.set_description("Val Loss: %.3f" % (total_loss / (i + 1)))
                out_bin = torch.argmax(out_bin, dim=1).cpu().numpy()
                metric_change.add_batch(out_bin, mask_bin.cpu().numpy())
            f1_change = metric_change.evaluate()
            Score_bin = 0.6*f1_change
            logger.info(
                "f1_change_Score:{}".format(f1_change))
            logger.info("Score_bin:{}".format(Score_bin))
            val_writer.add_scalar('Score', Score_bin, epoch)
            val_writer.add_scalar("loss", total_loss, epoch)
        logger.info("Val Loss:{}".format(total_loss / (i + 1)))

        Score_bin *= 100.0
        save_model_path = checkpoint_dir
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if Score_bin >= self.previous_best:
            if self.previous_best != 0:
                model_path = "%s/%s_%s_bin_%.2f.pth" % \
                             (save_model_path, self.args.model,
                              self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.state_dict(), "%s/%s_%s_bin_%.2f.pth" %
                       (save_model_path, self.args.model, self.args.backbone, Score_bin), _use_new_zipfile_serialization=False)
            self.previous_best = Score_bin
            # # 可视化
            for i in range(1):
                visualize_CD([img1[i].cpu().numpy(), img2[i].cpu().numpy()],
                             [mask_bin[i].cpu().numpy(), out_bin[i]],
                             imgs_dir, str(epoch)+"_"+id[i]+("_bin_%.2f" % (Score_bin)) + ".png")

            self.previous_best = Score_bin
            logger.info("epoch:{}  Score:{}".format(epoch, Score_bin))


if __name__ == "__main__":
    logger.info(args.exp_name)
    # -------------------- set directory for saving files -------------------------
    # 模型文件路径
    checkpoint_dir = os.path.join(
        'exp_result_original_data', args.exp_name, 'checkpoints')

    # 生成图像路径
    imgs_dir = os.path.join('exp_result_original_data',
                            args.exp_name, 'images')
    # 创建多级目录
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.isdir(imgs_dir):
        os.makedirs(imgs_dir)

    # 保存训练参数
    with open('{}/model_cfg.json'.format(logs_dir), 'w') as jfile:
        json.dump(vars(args), jfile, indent=2)

    # 设置可视化训练过程
    train_writer = SummaryWriter(os.path.join(
        logs_dir, 'runs', 'training'))
    val_writer = SummaryWriter(os.path.join(
        logs_dir, 'runs', 'val'))
    trainer = Trainer(args)

    for epoch in range(args.epochs):
        logger.info("\n")
        logger.info("\n==> Epoches %i/%i, learning rate = %.7f\t\t\t\t previous best = %.2f" %
                    (epoch, args.epochs, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        # 训练
        trainer.training()
        # 验证
        trainer.validation()
        # 更新学习率
        trainer.scheduler.step()
