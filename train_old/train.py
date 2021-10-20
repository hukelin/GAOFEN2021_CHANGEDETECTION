
from torchgeo1.models import FarSeg
from torchgeo.models import ChangeStarFarSeg
import json
import os
import shutil
import sys
import time
import warnings

from skimage.morphology import binary_dilation, disk
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
from datasets.change_detection import ChangeDetection
from loss import annealing_softmax_focalloss
from models.model_zoo import get_model
from models.pointrend import point_sample
from utils.image import add_image, visualize, visualize_edge
from utils.metric import F1_score
from utils.options import Options
from utils.palette import color_map
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer:
    def __init__(self, args):
        self.args = args
        # 加载数据
        trainset = ChangeDetection(
            root=args.data_root, mode="train", use_pseudo_label=args.use_pseudo_label)
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=4, drop_last=False)
        # *******************************模型选择*************************************
        # self.model = get_model(args.model, args.backbone, args.pretrained,
        #                        len(trainset.CLASSES)-1, args.lightweight)
        self.model = FarSeg(classes=2, backbone=args.backbone)
        # *******************************模型选择*************************************
        # 加载预训练模型
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(
                args.pretrain_from), strict=False)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        # *****************************损失函数的选择********************************
        # 二分类
        weight1 = torch.FloatTensor([0.54397636, 6.18487259]).cuda()
        weight2 = torch.FloatTensor([0.54902432, 5.59951011]).cuda()
        weight_bin = torch.FloatTensor([0.51030887, 24.75096457]).cuda()

        self.criterion1 = CrossEntropyLoss(weight=weight1)
        self.criterion2 = CrossEntropyLoss(weight=weight2)
        self.criterion_bin = CrossEntropyLoss(weight=weight_bin)
        # self.criterion_e1 = CrossEntropyLoss(
        #     torch.FloatTensor([0.50848837, 29.95205391]).cuda())
        # self.criterion_e2 = CrossEntropyLoss(
        #     torch.FloatTensor([0.5088186, 28.84916225]).cuda())
        # self.criterion_ec = CrossEntropyLoss(
        #     torch.FloatTensor([0.50165382, 151.66552401]).cuda())
        self.criterion = annealing_softmax_focalloss
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

        # self.model = self.model.cuda()  # 单GPU训练
        # print(next(self.model.parameters()).device)
        self.iters = 0
        # 迭代总次数
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best1 = 0.0
        self.previous_best2 = 0.0
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_sem = 0.0
        total_loss_bin = 0.0
        for i, (img1, img2, mask1, mask2, mask_bin, id) in enumerate(tbar):
            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2, mask_bin = mask1.cuda(), mask2.cuda(
            ), mask_bin.cuda()
            # 模型输出
            # out1, out2, out_bin, out_e1, out_e2, out_ec = self.model(
            #     img1, img2)
            out1, out2, out_bin = self.model(img1, img2)
            # print(out1.shape) torch.Size([bz, 2, 512, 512])
            loss1_1 = self.criterion(
                out1, mask1.long(), self.iters, self.total_iters)
            loss2_1 = self.criterion(
                out2, mask2.long(), self.iters, self.total_iters)
            loss_bin_1 = self.criterion(
                out_bin, mask_bin.long(), self.iters, self.total_iters)

            loss1_2 = self.criterion1(
                out1, mask1.long())
            loss2_2 = self.criterion2(
                out2, mask2.long())
            loss_bin_2 = self.criterion_bin(
                out_bin, mask_bin.long())
            # *****************添加边界约束****************
            # 二值化
            # out1 = torch.argmax(out1, dim=1).cpu().numpy()
            # out2 = torch.argmax(out2, dim=1).cpu().numpy()
            # out_bin = torch.argmax(out_bin, dim=1).cpu().numpy()
            # 转换边界
            # out_edge1 = torch.tensor(
            #     [binary_dilation(e1.squeeze(), selem=disk(2)) - e1 for e1 in out1])
            # out_edge2 = torch.tensor(
            #     [binary_dilation(e2.squeeze(), selem=disk(2)) - e2 for e2 in out2])
            # out_ec = torch.tensor(
            #     [binary_dilation(ec.squeeze(), selem=disk(2)) - ec for ec in out_bin])
            # loss_e1 = self.criterion_e1(out_e1, edge1.long())
            # loss_e2 = self.criterion_e2(out_e2, edge2.long())
            # loss_ec = self.criterion_ec(out_ec, edge_bin.long())

            loss1 = loss1_1 + loss1_2
            loss2 = loss2_1 + loss2_2
            loss_bin = loss_bin_1 + loss_bin_2

            loss1 = loss1.mean()
            loss2 = loss2.mean()
            loss_bin = loss_bin.mean()
            # 总loss
            # loss = loss_bin * 2 + loss1 + loss2
            loss = loss1 + loss2 + loss_bin

            total_loss_sem += loss1.item() + loss2.item()
            total_loss_bin += loss_bin.item()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iters += 1

            # lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            # self.optimizer.param_groups[0]["lr"] = lr
            # self.optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_sem / (i + 1), total_loss_bin / (i + 1)))
        train_writer.add_scalar("loss", total_loss_bin, epoch)

    def validation(self):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric1 = F1_score(num_classes=2)
        metric2 = F1_score(num_classes=2)
        metric_change = F1_score(num_classes=2)

        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for i, (img1, img2, mask1, mask2, mask_bin, id) in enumerate(tbar):
                img1, img2 = img1.cuda(), img2.cuda()
                # out1, out2, out_bin = self.model(
                #     img1, img2, self.args.tta)
                out1, out2, out_bin = self.model(img1, img2, True)
            #     mask1, mask2, mask_bin = mask1.cuda(), mask2.cuda(
            # ), mask_bin.cuda(), edge1.cuda(), edge2.cuda(), edge_bin.cuda()
                # out1, out2, out_bin, out_e1, out_e2, out_ec = self.model(
                #     img1, img2)
                out1 = torch.argmax(out1, dim=1).cpu().numpy()
                out2 = torch.argmax(out2, dim=1).cpu().numpy()
                out_bin = torch.argmax(out_bin, dim=1).cpu().numpy()

                # out_e1 = torch.argmax(out_e1, dim=1).cpu().numpy()
                # out_e2 = torch.argmax(out_e2, dim=1).cpu().numpy()
                # out_ec = torch.argmax(out_ec, dim=1).cpu().numpy()

                metric1.add_batch(out1, mask1.cpu().numpy())
                metric2.add_batch(out2, mask2.cpu().numpy())
                metric_change.add_batch(out_bin, mask_bin.cpu().numpy())
            f1_1 = metric1.evaluate()
            f1_2 = metric2.evaluate()
            f1_change = metric_change.evaluate()
            Score_seg = 0.2*(f1_1+f1_2)
            Score_bin = 0.6*f1_change
            Score = Score_seg + Score_bin
            # tbar.set_description("f1_1:%.6f  f1_2:%.6f  f1_change:%.6f  Score:%.6f" %
            #                      (f1_1*100.0, f1_2*100.0, f1_change*100.0, Score*100.0))
            logger.info(
                "f1_1_Score:{}  f1_2_Score:{}  f1_change_Score:{}".format(f1_1, f1_2, f1_change))
            logger.info(
                "Score_seg:{}  Score_bin:{}  Score:{}".format(Score_seg, Score_bin, Score))
            val_writer.add_scalar('Score', Score, epoch)
        if self.args.load_from:
            exit(0)
        Score_seg *= 100.0
        Score_bin *= 100.0
        Score *= 100.0
        save_model_path = checkpoint_dir
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        if Score_seg >= self.previous_best1:
            if self.previous_best1 != 0:
                model_path = "%s/%s_%s_seg_%.2f.pth" % \
                             (save_model_path, self.args.model,
                              self.args.backbone, self.previous_best1)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.state_dict(), "%s/%s_%s_seg_%.2f.pth" %
                       (save_model_path, self.args.model, self.args.backbone, Score_seg), _use_new_zipfile_serialization=False)
            self.previous_best1 = Score_seg
            visualize([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
                      [mask1[0].cpu().numpy(),
                       mask2[0].cpu().numpy()],
                      [out1[0], out2[0]],
                      [mask_bin[0].cpu().numpy(), out_bin[0]],
                      imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_seg_%.2f" % (Score_seg)) + ".png")
            # visualize_edge([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
            #                [edge1[0].cpu().numpy(),
            #                 edge2[0].cpu().numpy()],
            #                [out_e1[0], out_e2[0]],
            #                [edge_bin[0].cpu().numpy(), out_ec[0]],
            #                imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_edge_%.2f" % (Score_seg)) + ".png")

        if Score_bin >= self.previous_best2:
            if self.previous_best2 != 0:
                model_path = "%s/%s_%s_bin_%.2f.pth" % \
                             (save_model_path, self.args.model,
                              self.args.backbone, self.previous_best2)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.module.state_dict(), "%s/%s_%s_bin_%.2f.pth" %
                       (save_model_path, self.args.model, self.args.backbone, Score_bin), _use_new_zipfile_serialization=False)
            self.previous_best2 = Score_bin
            visualize([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
                      [mask1[0].cpu().numpy(),
                       mask2[0].cpu().numpy()],
                      [out1[0], out2[0]],
                      [mask_bin[0].cpu().numpy(), out_bin[0]],
                      imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_bin_%.2f" % (Score_bin)) + ".png")
            # visualize_edge([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
            #                [edge1[0].cpu().numpy(),
            #                 edge2[0].cpu().numpy()],
            #                [out_e1[0], out_e2[0]],
            #                [edge_bin[0].cpu().numpy(), out_ec[0]],
            #                imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_seg_%.2f" % (Score_seg)) + ".png")
        if Score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "%s/%s_%s_%.2f.pth" % \
                             (save_model_path, self.args.model,
                              self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.module.state_dict(), "%s/%s_%s_%.2f.pth" %
                       (save_model_path, self.args.model, self.args.backbone, Score), _use_new_zipfile_serialization=False)
            visualize([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
                      [mask1[0].cpu().numpy(),
                       mask2[0].cpu().numpy()],
                      [out1[0], out2[0]],
                      [mask_bin[0].cpu().numpy(), out_bin[0]],
                      imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_%.2f" % (Score)) + ".png")
            # visualize_edge([img1[0].cpu().numpy(), img2[0].cpu().numpy()],
            #                [edge1[0].cpu().numpy(),
            #                 edge2[0].cpu().numpy()],
            #                [out_e1[0], out_e2[0]],
            #                [edge_bin[0].cpu().numpy(), out_ec[0]],
            #                imgs_dir, str(epoch)+"_"+id[0].split(".")[0]+("_edge_%.2f" % (Score_seg)) + ".png")
            # if self.args.save_mask:
            # for i in range(out_bin.shape[0]):
            #     mask = Image.fromarray(
            #         out_bin[i].astype(np.uint8), mode="P")
            #     mask.putpalette(cmap)
            #     save_path = os.path.join(imgs_dir, "val")
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     mask.save(save_path + "/"+id[i])
            self.previous_best = Score
            logger.info("epoch:{}  Score:{}".format(epoch, Score))


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
