import os
from PIL import Image
from glob import glob
import shutil
# -----------------修改名称
# path = "/home/kelin/data/trainData"

# images = os.path.join(path, "images/*.png")
# images = glob(images)


# for i in range(1, int(len(images)/2)+1):
#     im1 = str(i) + "_1.png"
#     im2 = str(i) + "_2.png"
#     label1 = str(i) + "_1_label.png"
#     label2 = str(i) + "_2_label.png"
#     change = str(i)+"_change.png"
#     new_path = ["im1/"+im1, "im2/"+im2, "label1/" +
#                 label1, "label2/"+label2, "change/"+change]
#     old_path = ["images/"+im1, "images/"+im2,
#                 "gt/"+label1, "gt/"+label2, "gt/"+change]
#     for o, n in zip(old_path, new_path):
#         src = os.path.join(path, o)
#         dst = os.path.join(path, n)
#         new_path = os.path.dirname(dst)
#         if not os.path.exists(new_path):
#             os.makedirs(new_path)
#         shutil.move(src, dst)

# 交集取反
# import cv2
# import numpy as np
# label1 = cv2.imread("data/label1/1053.png")
# label2 = cv2.imread("data/label2/1053.png")
# label1[label1 == 255] = 1
# label2[label2 == 255] = 1
# l = (label1 + label2) - (label1 + label2).min() / \
#     ((label1 + label2).max() - (label1 + label2).min())
# # print(l.max())
# label = l - label1*label2
# # label = np.setxor1d(label1, label2)
# cv2.imwrite("label.png", label*255)
# print(np.array(label1))

# 读取数据
l_cd = "/home/kelin/data/CD_data"

tr = os.path.exists(l_cd,"train")
