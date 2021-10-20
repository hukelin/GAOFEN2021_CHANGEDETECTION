import torch.nn as nn
from sklearn.metrics import f1_score
import math
import numpy as np


def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class F1_score:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 筛选标签 (262144,)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # minlength---bin的数量
        # 0 0 TN--0
        # 0 1 FP--1
        # 1 0 FN--2
        # 1 1 TP--3
        hist = np.bincount(self.num_classes * label_true[mask].astype(
            int) + label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    # 获取混淆矩阵

    def add_batch(self, predictions, gts):
        # 遍历预测值和真实值 predictions:(bz, 512, 512) gts:(bz,512, 512)
        # lp:(512,512)  lt:(512,512)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        # self.hist[0][0]--TN
        # self.hist[0][1]--FP
        # self.hist[1][0]--FN
        # self.hist[1][1]--TP
        # self.hist.sum(0) (TN+FN,FP+TP) (预测为0的个数，预测为1的个数)
        # self.hist.sum(1) (TN+FP,FN+TP) (实际为0的个数，实际为1的个数)
        # print(self.hist[0][0])
        # print(self.hist[0][1])
        # print(self.hist[1][0])
        # print(self.hist[1][1])
        # 计算F1-Score
        # precision = TP/(TP + FP)
        precision = self.hist[1][1] / (self.hist[1][1]+self.hist[0][1])
        print(precision)
        # recall = TP/(TP + FN)
        recall = self.hist[1][1] / (self.hist[1][1]+self.hist[1][0])
        print(recall)
        # F1-Score
        F1_Score = (2*precision*recall) / (precision+recall)

        return F1_Score


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 筛选标签
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # minlength---bin的数量
        # 0 0 --0
        # 0 1 --1
        # 1 0 --2
        # 1 1 --3
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    # 获取混淆矩阵

    def add_batch(self, predictions, gts):
        # 遍历预测值和真实值 lp:(1, 512, 512) lt:(512, 512)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        confusion_matrix = np.zeros((2, 2))
        # TN
        confusion_matrix[0][0] = self.hist[0][0]
        # FP
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        # FN
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        # TP
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()
        #
        # confusion_matrix.sum(0) (TN+FN,FP+TP) (预测为0的个数，预测为1的个数)
        # confusion_matrix.sum(1) (TN+FP,FN+TP) (实际为0的个数，实际为1的个数)
        # np.diag(confusion_matrix) (TN,TP)
        # 0:TN / FN+FP+TN
        # 1:TP / FP+FN+TP
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)
        # hist = self.hist.copy()
        # hist[0][0] = 0
        # kappa = cal_kappa(hist)
        # sek = kappa * math.exp(iou[1] - 1)

        # score = 0.3 * miou + 0.7 * sek

        # return score, miou, sek
        return iou, miou

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)
