from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map
from utils.metric import IOUandSek

import numpy as np
import os
from PIL import Image
import shutil
import torch
# import torchcontrib
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

class Trainer:
    def __init__(self, args):
        self.args = args

        trainset = ChangeDetection(
            root=args.data_root, mode="train", use_pseudo_label=args.use_pseudo_label)
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=4, drop_last=False)

        self.model = get_model(args.model, args.backbone, args.pretrained,
                               len(trainset.CLASSES)-1, args.lightweight)
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(
                args.pretrain_from), strict=False)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        if args.use_pseudo_label:
            weight = torch.FloatTensor([1, 1, 1, 1, 1, 1]).cuda()
        else:
            weight = torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda()
        # 多分类
        self.criterion = CrossEntropyLoss(ignore_index=-1, weight=weight)
        # 二分类
        self.criterion_bin = BCELoss(reduction='none')

        self.optimizer = Adam([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": args.lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": args.lr * 10.0}],
                              lr=args.lr, weight_decay=args.weight_decay)

        self.model = DataParallel(self.model).cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_sem = 0.0
        total_loss_bin = 0.0

        for i, (img1, img2, mask1, mask2, mask_bin) in enumerate(tbar):
            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2 = mask1.cuda(), mask2.cuda()
            mask_bin = mask_bin.cuda()
            print("img1 shape", img1.shape)
            print("img2 shape", img1.shape)
            print("mask1 shape", mask1.shape)
            print("mask2 shape", mask2.shape)
            print("mask_bin shape", mask_bin.shape)
            out1, out2, out_bin = self.model(img1, img2)
            print("\n")
            loss1 = self.criterion(out1, mask1 - 1)
            loss2 = self.criterion(out2, mask2 - 1)
            print("out1 shape",out_bin.shape)
            print("out2 shape",out_bin.shape)
            print("out_bin shape",out_bin.shape)
            # img1 shape torch.Size([8, 3, 512, 512])
            # img2 shape torch.Size([8, 3, 512, 512])
            # mask1 shape torch.Size([8, 512, 512])
            # mask2 shape torch.Size([8, 512, 512])
            # mask_bin shape torch.Size([8, 512, 512])
            # out1 shape torch.Size([8, 512, 512])
            # out2 shape torch.Size([8, 512, 512])
            # out_bin shape torch.Size([8, 512, 512])
            print(mask_bin)
            loss_bin = self.criterion_bin(out_bin, mask_bin)
            # 将变化区域loss权重设为2
            loss_bin[mask_bin == 0] *= 2
            
            loss_bin = loss_bin.mean()
            # 总loss
            loss = loss_bin * 2 + loss1 + loss2
            print(loss)
            # sys.exit()
            total_loss_sem += loss1.item() + loss2.item()
            total_loss_bin += loss_bin.item()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iters += 1
            lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_sem / (i + 1), total_loss_bin / (i + 1)))

    def validation(self):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection.CLASSES))
        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, img2, mask1, mask2, id in tbar:
                img1, img2 = img1.cuda(), img2.cuda()

                out1, out2, out_bin = self.model(img1, img2, self.args.tta)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                out_bin = (out_bin > 0.5).cpu().numpy().astype(np.uint8)
                # out_bin输出为1表示-未变化区域
                out1[out_bin == 1] = 0
                out2[out_bin == 1] = 0

                if self.args.save_mask:
                    for i in range(out1.shape[0]):
                        mask = Image.fromarray(
                            out1[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/train/im1/" + id[i])

                        mask = Image.fromarray(
                            out2[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/train/im2/" + id[i])

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())

                score, miou, sek = metric.evaluate()

                tbar.set_description("Score: %.2f, IOU: %.2f, SeK: %.2f" % (
                    score * 100.0, miou * 100.0, sek * 100.0))

        if self.args.load_from:
            exit(0)

        score *= 100.0
        if score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/models/change_detection/%s_%s_%.2f.pth" % \
                             (self.args.model, self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)

            torch.save(self.model.module.state_dict(), "outdir/models/change_detection/%s_%s_%.2f.pth" %
                       (self.args.model, self.args.backbone, score), _use_new_zipfile_serialization=False)
            self.previous_best = score


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.2f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training()
        trainer.validation()
