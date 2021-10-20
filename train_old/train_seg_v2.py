
from model import U2NET
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from torchgeo1.models import FarSeg
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
# import torchcontrib
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
import loss
from datasets.segmentation import Segmentation
from loss import annealing_softmax_focalloss
from models.model_zoo import get_model
from models.pointrend import point_sample
from utils.image import add_image, visualize, visualize_CD, visualize_edge, visualize_seg
from utils.metric import F1_score
from utils.options import Options
from utils.palette import color_map
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from loss import bce_dice_loss

def layer_ce_loss(y_pred, y_true, l_weight=1.0):
    label = y_true.detach().cpu().numpy()
    # 统计类别数量
    label_num = np.bincount(label.flatten())
    weights = label_num.sum() / (label_num * 2)
    if len(weights) == 1:
        weights = np.append(weights, 0.)
    weights = torch.FloatTensor(weights).cuda()
    loss = CrossEntropyLoss(weight=weights, reduction='mean')(y_pred, y_true)
    return l_weight*loss


class Trainer:
    def __init__(self, args):
        self.args = args
        # 加载数据
        trainset = Segmentation(
            root=args.data_root, mode="train")
        valset = Segmentation(args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=4, drop_last=False)
        # *******************************模型选择*************************************
        self.model = U2NET(3, 2)
        # *******************************模型选择*************************************
        # 加载预训练模型
        if args.pretrain_from:
            print("hello")
            self.model.load_state_dict(
                torch.load(args.pretrain_from), strict=True)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        # *****************************损失函数的选择********************************
        # 二分类
        # weight_seg = torch.FloatTensor([0.54648868, 5.87765307]).cuda()
        # self.criterion_seg = CrossEntropyLoss(weight=weight_seg)
        self.criterion_seg = bce_dice_loss
        # self.criterion = annealing_softmax_focalloss
        # *****************************损失函数的选择********************************
        #
        # param = [name for name, param in self.model.named_parameters()]
        # print(param)
        # sys.exit()
        # 设置优化器
        self.optimizer = Adam([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": args.lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": args.lr * 10.0}],
                              lr=args.lr, weight_decay=args.weight_decay)
        # 调整学习率
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=3,  # T_0就是初始restart的epoch数目
            T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 + T_0 * T_mult
            eta_min=1e-6  # 最低学习率
        )
        # 模型并行训练
        self.model = DataParallel(self.model).cuda()
        # self.model = self.model.cuda()  # 单GPU训练
        # print(next(self.model.parameters()).device)
        self.iters = 0
        # 迭代总次数
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        loss_seg1 = 0.0
        loss_seg2 = 0.0
        for i, (img, label, id) in enumerate(tbar):
            img = img.cuda()
            mask_seg = label.cuda()
            # 模型输出
            out_seg = self.model(img)
            # out_seg = self.model(img)
            # loss_seg1 = self.criterion_seg(out_seg.squeeze(), mask_seg.long())
            # loss_seg2 = self.criterion(
            #     out_seg.squeeze(), mask_seg.long(), self.iters, self.total_iters)
            l_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
            loss_seg1 = sum([self.criterion_seg(preds, mask_seg.long(), l_w)
                             for preds, l_w in zip(out_seg, l_weight)])
            loss = loss_seg1+loss_seg2
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

    def validation(self):
        total_loss = 0.0
        loss_seg1 = 0.0
        loss_seg2 = 0.0
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric_seg = F1_score(num_classes=2)
        with torch.no_grad():
            for i, (img, label, id) in enumerate(tbar):
                img = img.cuda()
                mask_seg = label.cuda()
                # 模型输出
                out_seg = self.model(img)
                l_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
                loss_seg1 = sum([self.criterion_seg(preds, mask_seg.long(), l_w)
                                 for preds, l_w in zip(out_seg, l_weight)])
                out_seg = out_seg[0]
                # loss_seg2 = self.criterion(
                #     out_seg.squeeze(), mask_seg.long(), self.iters, self.total_iters)
                loss = loss_seg1+loss_seg2
                loss = loss.mean()
                total_loss += loss.item()
                tbar.set_description("Val Loss: %.3f" % (total_loss / (i + 1)))
                out_seg = torch.argmax(out_seg, dim=1).cpu().numpy()
                metric_seg.add_batch(out_seg, mask_seg.cpu().numpy())
            f1_seg = metric_seg.evaluate()
            Score_seg = 0.1*f1_seg
            logger.info("f1_seg_Score:{}".format(f1_seg))
            logger.info("Score_seg:{}".format(Score_seg))
            val_writer.add_scalar('Score', Score_seg, epoch)
        if self.args.load_from:
            exit(0)

        Score_seg *= 100.0
        save_model_path = checkpoint_dir
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if Score_seg >= self.previous_best:
            if self.previous_best != 0:
                model_path = "%s/%s_%s_seg_%.2f.pth" % \
                             (save_model_path, self.args.model,
                              self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.module.state_dict(), "%s/%s_%s_seg_%.2f.pth" %
                       (save_model_path, self.args.model, self.args.backbone, Score_seg), _use_new_zipfile_serialization=False)
            self.previous_best = Score_seg
            # 可视化
            visualize_seg([img[0].cpu().numpy(), img[1].cpu().numpy()],
                          [mask_seg[0].cpu().numpy(), mask_seg[1].cpu().numpy()],
                          [out_seg[0], out_seg[1]],
                          imgs_dir, str(epoch)+"_"+id[0]+("_seg_%.2f" % (Score_seg)) + ".png")

            self.previous_best = Score_seg
            logger.info("epoch:{}  Score:{}".format(epoch, Score_seg))


if __name__ == "__main__":
    # 加载训练参数
    args = Options().parse()
    # 设置实验保存文件夹
    time_str = time.strftime('%Y-%m-%d-%H:%M')
    args.exp_name = '{}_{}_{}_{}'.format(
        args.model, args.backbone, args.exp_name, time_str)
    logger.info(args.exp_name)
    # -------------------- set directory for saving files -------------------------
    # 模型文件路径
    checkpoint_dir = os.path.join(
        'exp_result', args.exp_name, 'checkpoints')
    # 日志记录路径
    logs_dir = os.path.join('exp_result', args.exp_name, 'logs')
    # 生成图像路径
    imgs_dir = os.path.join('exp_result', args.exp_name, 'images')
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

    if args.load_from:
        trainer.validation()
    # 日志记录文件
    logger.add("{}/val_log.log".format(logs_dir))

    for epoch in range(args.epochs):
        logger.info("\n")
        logger.info("\n==> Epoches %i/%i, learning rate = %.6f\t\t\t\t previous best = %.2f" %
                    (epoch, args.epochs, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        # 训练
        trainer.training()
        # 验证
        trainer.validation()
        # 更新学习率
        trainer.scheduler.step()
