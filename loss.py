import ever.module as M
import math
import torch.nn.functional as F
import torch
import torch.nn as nn

import numpy as np


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


def softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0, normalize=False):
    """

    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar

    Returns:

    """
    losses = F.cross_entropy(
        y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(
            valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(
            modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / \
        (valid_mask.sum() + p.size(0))

    return losses


def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)


def poly_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = (1 - _t / _t_max) ** 0.9
    return upper_bound + factor * (lower_bound - upper_bound)


def linear_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = 1 - _t / _t_max
    return upper_bound + factor * (lower_bound - upper_bound)


def annealing_softmax_focalloss(y_pred, y_true, t, t_max, ignore_index=255, gamma=2.0,
                                annealing_function=cosine_annealing):
    losses = F.cross_entropy(
        y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(
            valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(
            modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        normalizer = losses.sum() / (losses * modulating_factor).sum()
        scales = modulating_factor * normalizer
    if t > t_max:
        scale = scales
    else:
        scale = annealing_function(1, scales, t, t_max)
    losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
    return losses


def misc_info(device):
    loss_dict = dict()
    mem = torch.cuda.max_memory_allocated() // 1024 // 1024
    loss_dict['mem'] = torch.from_numpy(
        np.array([mem], dtype=np.float32)).to(device)
    return loss_dict


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, loss_config, prefix=''):
    loss_dict = dict()
    # 设置权重
    weight = loss_config.get('weight', 1.0)
    if loss_config.get('dice', False):
        loss_dict[f'{prefix}dice_loss'] = weight * M.loss.dice_loss_with_logits(
            y_pred, y_true, ignore_index=loss_config['ignore_index'])
    #
    if loss_config.get('bce', False):
        loss_dict[f'{prefix}bce_loss'] = weight * M.loss.binary_cross_entropy_with_logits(y_pred, y_true,
                                                                                          ignore_index=loss_config['ignore_index'])

    return loss_dict


def ce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, prefix=''):
    loss_dict = dict()
    loss_dict[f'{prefix}dice_loss'] = M.loss.dice_loss_with_logits(
        y_pred, y_true)

    loss_dict[f'{prefix}bce_loss'] = M.loss.binary_cross_entropy_with_logits(
        y_pred, y_true)

    return loss_dict


def symmetry_loss(positive_mask,
                  change_y1vy2_logit,
                  change_y2vy1_logit,
                  loss_config
                  ):
    total_loss = dict()
    if change_y1vy2_logit is not None:
        total_loss.update(
            bce_dice_loss(positive_mask, change_y1vy2_logit, loss_config, 'c12'))
    if change_y2vy1_logit is not None:
        total_loss.update(
            bce_dice_loss(positive_mask, change_y2vy1_logit, loss_config, 'c21'))

    return total_loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        #
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        #
        ph, pw = score.size(2), score.size(3)
        #
        h, w = target.size(1), target.size(2)
        #
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        #
        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        #
        ph, pw = score.size(2), score.size(3)
        #
        h, w = target.size(1), target.size(2)
        #
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        #
        pred = F.softmax(score, dim=1)
        #
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        #
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        score = [score]
        weights = [1]
        # print(len(weights))
        # print(len(score))
        assert len(weights) == len(score)
        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss
