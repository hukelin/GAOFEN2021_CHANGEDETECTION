import numpy as np
import random
import cv2
import torch
from torch.nn import functional as F


class MultiScale(object):
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        # random.randint 返回一个随机整型数，范围从低（包括）到高（不包括）
        rand_scale = 1 + random.randint(0, 16) / 10.0
        img, label = multi_scale_aug(
            img, label, rand_scale=rand_scale)
        return {"img": img, "label": label}


def multi_scale_inference(model, image, scales=[1.0, 1.25, 1.5, 1.75, 2.0]):
    # 获取图像尺寸 torch.Size([1, 3, 512, 512])
    batch, _, ori_height, ori_width = image.size()
    assert batch == 1, "only supporting batchsize 1."
    # tensor-->numpy 转换维度 (c,h,w)-->(h,w,c) (512, 512, 3)
    image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()
    # 步长
    stride_h = 256
    stride_w = 256
    # 生成和预测图像形状一致的零张量 torch.Size([1, 2, 512, 512])
    final_pred = torch.zeros([1, 2, ori_height, ori_width]).cuda()
    # 填充值
    padvalue = (0.0, 0.0, 0.0)
    # 遍历多尺度
    for scale in scales:
        # 生成多尺度输入图像 image--(h,w,c)
        new_img = multi_scale_aug(
            image=image, rand_scale=scale, rand_crop=False)
        # 获取新图像的长宽
        height, width = new_img.shape[:-1]
        # 填充，小尺度
        if max(height, width) <= np.min((512, 512)):
            # 填充
            new_img = pad_img(new_img, height, width, (512, 512), padvalue)
            # (h,w,c)-->(c,h,w) (3, 512, 512)
            new_img = new_img.transpose((2, 0, 1))
            # 增加维度 (1, 3, 512, 512)
            new_img = np.expand_dims(new_img, axis=0)
            # numpy-->tensor
            new_img = torch.from_numpy(new_img).cuda()
            # 输出预测结果 torch.Size([1, 2, 512, 512])
            preds = model(new_img)
            # 输出结果
            preds = preds[:, :, 0:height, 0:width]
        # 大尺度
        else:
            if height < 512 or width < 512:
                new_img = pad_img(new_img, height, width,
                                  (512, 512), padvalue)
            # 获取新图像大小
            new_h, new_w = new_img.shape[:2]
            # np.ceil 计算各元素的ceiling，对元素向上取整
            rows = np.int(np.ceil(1.0 * (new_h - 512) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w - 512) / stride_w)) + 1
            #
            preds = torch.zeros([1, 2, new_h, new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()
            # 遍历行列数
            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    # 取最小
                    h1 = min(h0 + 512, new_h)
                    w1 = min(w0 + 512, new_w)
                    # 裁剪图像 (512, 512, 3)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    if h1 == new_h or w1 == new_w:
                        # 填充图像
                        crop_img = pad_img(crop_img, h1-h0, w1-w0, (512, 512), padvalue)
                    # (h,w,c)-->(c,h,w)
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img).cuda()
                    pred = model(crop_img)  # torch.Size([1, 2, 512, 512])
                    #
                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                    count[:, :, h0:h1, w0:w1] += 1
            preds = preds / count
            preds = preds[:, :, :height, :width]
        preds = F.interpolate(preds, (ori_height, ori_width),
                              mode='bilinear', align_corners=True)
        final_pred += preds
    return final_pred


def pad_img(img, h, w, size, padvalue):
    pad_img = img.copy()
    #
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)

    if pad_h > 0 or pad_w > 0:
        # cv2.copyMakeBorder()用来给图片添加边框
        # top, bottom, left, right：上下左右要扩展的像素数
        pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0,
                                     pad_w, cv2.BORDER_CONSTANT,
                                     value=padvalue)

    return pad_img


def rand_crop(img, label):
    h, w = img.shape[:2]
    crop_size = (512, 512)
    # 填充图像
    img = pad_img(img, h, w, crop_size, (0.0, 0.0, 0.0))

    label = pad_img(label, h, w, crop_size, (0.0, 0.0, 0.0))
    new_h, new_w = label.shape
    x = random.randint(0, new_w - crop_size[1])
    y = random.randint(0, new_h - crop_size[0])
    # 转换到原始大小
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    label = label[y:y+crop_size[0], x:x+crop_size[1]]

    return img, label


def multi_scale_aug(image, label=None,
                    rand_scale=1, rand_crop=True):
    long_size = np.int(512 * rand_scale + 0.5)  # [512,640,]
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)
    # (512, 512, 3) (640, 640, 3)
    image = cv2.resize(image, (new_w, new_h),
                       interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (new_w, new_h),
                           interpolation=cv2.INTER_NEAREST)
    else:
        return image

    if rand_crop:
        image, label = rand_crop(image, label)


class RandomFlip(object):
    def __call__(self, sample):
        # Random Left or Right Flip
        img, label = sample['img'], sample['label']
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        else:
            img = np.flipud(img)
            label = np.flipud(label)
        return {"img": img,  "label": label}
