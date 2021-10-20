import os
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# import cv2
import numpy as np
import torch
from matplotlib import colors
from PIL import Image

# import os

# import kornia as kn


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def restore_rgb(config, I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if len(I) > 3 and not type(I) == np.ndarray:
        I = np.array(I)
        I = I[:, :, :, 0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i, ...]
            x = np.array(x, dtype=np.float32)
            x = x * config[2] + config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            #
            x = image_normalization(x)
            I[i, :, :, :] = x
    elif len(I.shape) == 3 and I.shape[-1] == 3:
        I = np.array(I, dtype=np.float32)
        I = I*config[2] + config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    return I


def stretchImg(img, lower_percent=0.6, higher_percent=99.4):
    # 百分比拉伸
    # print(data)
    # sys.exit()
    if img is not None:
        n = img.shape[2]
        out = np.zeros_like(img, dtype=np.uint8)
        for i in range(n):
            a = 0
            b = 255
            c = np.percentile(img[:, :, i], lower_percent)
            d = np.percentile(img[:, :, i], higher_percent)
            t = a + (img[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
            outImg = np.uint8(out)
        return outImg


def visualize(img: List, mask: List, seg: List, change: List, visualize_path: str, id: int):
    # mean1 = [0.3542, 0.3497, 0.3170]
    # std1 = [0.1852, 0.1598, 0.1612]
    # mean2 = [0.3155, 0.3199, 0.2932]
    # std2 = [0.1981, 0.1773, 0.1904]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all',
                            sharey='all', figsize=(30, 16))
    for i in range(2):
        # print(img[i].shape)
        img_ = img[i].transpose((1, 2, 0))
        # if i == 0:
        #     img_ = np.clip(img_*std1 + mean1, 0, 1)
        # if i == 1:
        #     img_ = np.clip(img_*std2 + mean2, 0, 1)
        # img_ = np.clip(img_*std + mean, 0, 1)
        axs[i][0].imshow(img_)
        axs[i][1].imshow(mask[i].squeeze(), cmap='gray')
        axs[i][2].imshow(seg[i].squeeze(), cmap='gray')
        axs[i][3].imshow(change[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "seg")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_CD(img: List, change: List, visualize_path: str, id: int):
    mean1 = (90.3236, 89.1732, 80.8296)
    std1 = (47.2191, 40.7412, 41.1059)
    mean2 = (80.4520, 81.5796, 74.7567)
    std2 = (50.5237, 45.2135, 48.5634)
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all',
                            sharey='all', figsize=(16, 16))
    for i in range(2):
        # print(img[i].shape)
        img_ = img[i].transpose((1, 2, 0))
        if i == 0:
            img_ = img_*std1 + mean1
        if i == 1:
            img_ = img_*std2 + mean2
        axs[i][0].imshow(np.uint8(img_))
        # axs[i][1].imshow(seg[i].squeeze(), cmap='gray')
        axs[i][1].imshow(change[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "change")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_CD2(img: List, seg: List, change: List, visualize_path: str, id: int):
    mean1 = (90.3236, 89.1732, 80.8296)
    std1 = (47.2191, 40.7412, 41.1059)
    mean2 = (80.4520, 81.5796, 74.7567)
    std2 = (50.5237, 45.2135, 48.5634)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex='all',
                            sharey='all', figsize=(24, 16))
    for i in range(2):
        # print(img[i].shape)
        img_ = img[i].transpose((1, 2, 0))
        if i == 0:
            img_ = img_*std1 + mean1
        if i == 1:
            img_ = img_*std2 + mean2
        # img_ = np.clip(img_*std + mean, 0, 1)
        axs[i][0].imshow(img_)
        axs[i][1].imshow(seg[i].squeeze(), cmap='gray')
        axs[i][2].imshow(change[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "change")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_edge(img: List, mask: List, seg: List, change: List, visualize_path: str, id: int):
    # mean1 = [0.3542, 0.3497, 0.3170]
    # std1 = [0.1852, 0.1598, 0.1612]
    # mean2 = [0.3155, 0.3199, 0.2932]
    # std2 = [0.1981, 0.1773, 0.1904]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all',
                            sharey='all', figsize=(30, 16))
    for i in range(2):
        # print(img[i].shape)
        img_ = img[i].transpose((1, 2, 0))
        # if i == 0:
        #     img_ = np.clip(img_*std1 + mean1, 0, 1)
        # if i == 1:
        #     img_ = np.clip(img_*std2 + mean2, 0, 1)
        img_ = np.clip(img_*std + mean, 0, 1)
        axs[i][0].imshow(img_)
        axs[i][1].imshow(mask[i].squeeze(), cmap='gray')
        axs[i][2].imshow(seg[i].squeeze(), cmap='gray')
        axs[i][3].imshow(change[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "edge")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_seg(img: List, mask: List, seg: List, visualize_path: str, id: int):
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex='all',
                            sharey='all', figsize=(24, 16))
    for i in range(2):
        img_ = img[i].transpose((1, 2, 0))
        img_ = np.clip(img_, 0, 1)
        axs[i][0].imshow(img_)
        axs[i][1].imshow(mask[i].squeeze(), cmap='gray')
        axs[i][2].imshow(seg[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "seg")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_edge(img: List, mask: List, seg: List, change: List, visualize_path: str, id: int):
    # mean1 = [0.3542, 0.3497, 0.3170]
    # std1 = [0.1852, 0.1598, 0.1612]
    # mean2 = [0.3155, 0.3199, 0.2932]
    # std2 = [0.1981, 0.1773, 0.1904]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 可视化输出图像
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all',
                            sharey='all', figsize=(30, 16))
    for i in range(2):
        # print(img[i].shape)
        img_ = img[i].transpose((1, 2, 0))
        # if i == 0:
        #     img_ = np.clip(img_*std1 + mean1, 0, 1)
        # if i == 1:
        #     img_ = np.clip(img_*std2 + mean2, 0, 1)
        img_ = np.clip(img_*std + mean, 0, 1)
        axs[i][0].imshow(img_)
        axs[i][1].imshow(mask[i].squeeze(), cmap='gray')
        axs[i][2].imshow(seg[i].squeeze(), cmap='gray')
        axs[i][3].imshow(change[i].squeeze(), cmap='gray')
    plt.tight_layout()
    out_path = os.path.join(visualize_path, "edge")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig("{}/{}".format(out_path, id))
    plt.close()


def visualize_change(mask, seg, visualize_path, id):
    # 可视化变化检测结果
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='all',
                            sharey='all', figsize=(50, 50))
    for i in range(2):
        img = stretchImg(img[i])
        axs[i][0].imshow(img)
        axs[i][1].imshow(mask[i], cmap='gray')
        axs[i][2].imshow(seg[i], cmap='gray')

    plt.tight_layout()
    plt.savefig("{}/{}.png".format(visualize_path), id+"_label")


def display_input_batch(tensor, display_indices=0, brightness_factor=1):

    # extract display channels
    # tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    # scale image
    # 限制输出范围
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def mycmap():
    cmap = colors.ListedColormap(['#000000',
                                  '#FFFFFF'])
    return cmap


def display_label_batch(tensor):
    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]
    # colorize labels
    cmap = mycmap()
    imgs = []
    # tensor.shape[0]---batch_size
    # print(tensor.shape)
    for s in range(tensor.shape[0]):
        im = tensor[s, :, :].cpu().numpy()
        im = cmap(im)[:, :, 0:3]
        # 换轴
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def add_image(writer, image, prediction, target, global_step):
    # write some example images to tensorboard every n steps
    if global_step > 0:
        # 增加图像至tensorboard中
        # writer.add_images("val/input", image[:, 0:3, :, :],
        #                   global_step=global_step)
        writer.flush()
        # 显示图像
        imgs = display_input_batch(image)
        writer.add_images("val/input", imgs, global_step=global_step)

        # 显示真实标签
        imgs = display_label_batch(target)
        writer.add_images("val/ground_truth", imgs,
                          global_step=global_step)

        # 显示预测结果
        imgs = display_label_batch(prediction)
        writer.add_images("val/prediction", imgs,
                          global_step=global_step)


if __name__ == "__main__":
    label = Image.open("/home/kelin/data/train/label1/2000.png")
    label = torch.tensor(np.array(label))
    imgs = display_label_batch(label)
    print(imgs.shape)
