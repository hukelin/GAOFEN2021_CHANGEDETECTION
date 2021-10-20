import numpy as np
from PIL import Image
import random


# class RandomFlipOrRotate(object):
#     def __call__(self, sample):
#         img1, img2, mask1, mask2, mask_bin = \
#             sample['img1'], sample['img2'], sample['mask1'], sample['mask2'], sample['mask_bin']
#         # random() 方法返回随机生成的一个实数，它在[0,1)范围内
#         rand = random.random()
#         if rand < 1 / 6:
#             img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
#             mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
#             img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
#             mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
#             mask_bin = mask_bin.transpose(Image.FLIP_LEFT_RIGHT)

#         elif rand < 2 / 6:
#             img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
#             mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
#             img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
#             mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
#             mask_bin = mask_bin.transpose(Image.FLIP_TOP_BOTTOM)

#         elif rand < 3 / 6:
#             img1 = img1.transpose(Image.ROTATE_90)
#             mask1 = mask1.transpose(Image.ROTATE_90)
#             img2 = img2.transpose(Image.ROTATE_90)
#             mask2 = mask2.transpose(Image.ROTATE_90)
#             mask_bin = mask_bin.transpose(Image.ROTATE_90)

#         elif rand < 4 / 6:
#             img1 = img1.transpose(Image.ROTATE_180)
#             mask1 = mask1.transpose(Image.ROTATE_180)
#             img2 = img2.transpose(Image.ROTATE_180)
#             mask2 = mask2.transpose(Image.ROTATE_180)
#             mask_bin = mask_bin.transpose(Image.ROTATE_180)

#         elif rand < 5 / 6:
#             img1 = img1.transpose(Image.ROTATE_270)
#             mask1 = mask1.transpose(Image.ROTATE_270)
#             img2 = img2.transpose(Image.ROTATE_270)
#             mask2 = mask2.transpose(Image.ROTATE_270)
#             mask_bin = mask_bin.transpose(Image.ROTATE_270)

#         return {'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2, 'mask_bin': mask_bin}

class RandomFlipOrRotate(object):
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        # random() 方法返回随机生成的一个实数，它在[0,1)范围内
        rand = random.random()
        if rand < 1 / 6:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        elif rand < 2 / 6:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        elif rand < 3 / 6:
            img = img.transpose(Image.ROTATE_90)
            label = label.transpose(Image.ROTATE_90)

        elif rand < 4 / 6:
            img = img.transpose(Image.ROTATE_180)
            label = label.transpose(Image.ROTATE_180)

        elif rand < 5 / 6:
            img = img.transpose(Image.ROTATE_270)
            label = label.transpose(Image.ROTATE_270)

        return {'img': img, 'label': label}
#
# class RandomFlipOrRotate(object):
#     def __call__(self, sample):
#         img1, img2, mask1, mask2, mask_bin, edge1, edge2, edge_bin = sample['img1'], sample[
#             'img2'], sample['mask1'], sample['mask2'], sample['mask_bin'], sample['edge1'], sample['edge2'], sample['edge_bin']
#         # random() 方法返回随机生成的一个实数，它在[0,1)范围内
#         rand = random.random()
#         if rand < 1 / 6:
#             img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
#             mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
#             img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
#             mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
#             mask_bin = mask_bin.transpose(Image.FLIP_LEFT_RIGHT)
#             edge1 = edge1.transpose(Image.FLIP_LEFT_RIGHT)
#             edge2 = edge2.transpose(Image.FLIP_LEFT_RIGHT)
#             edge_bin = edge_bin.transpose(Image.FLIP_LEFT_RIGHT)
#         elif rand < 2 / 6:
#             img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
#             mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
#             img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
#             mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
#             mask_bin = mask_bin.transpose(Image.FLIP_TOP_BOTTOM)
#             edge1 = edge1.transpose(Image.FLIP_TOP_BOTTOM)
#             edge2 = edge2.transpose(Image.FLIP_TOP_BOTTOM)
#             edge_bin = edge_bin.transpose(Image.FLIP_TOP_BOTTOM)
#         elif rand < 3 / 6:
#             img1 = img1.transpose(Image.ROTATE_90)
#             mask1 = mask1.transpose(Image.ROTATE_90)
#             img2 = img2.transpose(Image.ROTATE_90)
#             mask2 = mask2.transpose(Image.ROTATE_90)
#             mask_bin = mask_bin.transpose(Image.ROTATE_90)
#             edge1 = edge1.transpose(Image.ROTATE_90)
#             edge2 = edge2.transpose(Image.ROTATE_90)
#             edge_bin = edge_bin.transpose(Image.ROTATE_90)
#         elif rand < 4 / 6:
#             img1 = img1.transpose(Image.ROTATE_180)
#             mask1 = mask1.transpose(Image.ROTATE_180)
#             img2 = img2.transpose(Image.ROTATE_180)
#             mask2 = mask2.transpose(Image.ROTATE_180)
#             mask_bin = mask_bin.transpose(Image.ROTATE_180)
#             edge1 = edge1.transpose(Image.ROTATE_180)
#             edge2 = edge2.transpose(Image.ROTATE_180)
#             edge_bin = edge_bin.transpose(Image.ROTATE_180)
#         elif rand < 5 / 6:
#             img1 = img1.transpose(Image.ROTATE_270)
#             mask1 = mask1.transpose(Image.ROTATE_270)
#             img2 = img2.transpose(Image.ROTATE_270)
#             mask2 = mask2.transpose(Image.ROTATE_270)
#             mask_bin = mask_bin.transpose(Image.ROTATE_270)
#             edge1 = edge1.transpose(Image.ROTATE_270)
#             edge2 = edge2.transpose(Image.ROTATE_270)
#             edge_bin = edge_bin.transpose(Image.ROTATE_270)
#         return {'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
#                 'mask_bin': mask_bin, 'edge1': edge1, 'edge2': edge2, 'edge_bin': edge_bin}
