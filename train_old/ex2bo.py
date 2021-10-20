import cv2
import glob
import os
import numpy as np

from skimage.morphology import binary_dilation, disk


# class Extent2Boundary():
#     """
#     Adds boundary mask from extent mask using binary dilation
#     """

#     def __init__(self, extent_feature: Tuple[FeatureType, str],
#                  boundary_feature: Tuple[FeatureType, str], structure: np.ndarray = None):
#         self.extent_feature = next(self._parse_features(extent_feature)())
#         self.boundary_feature = next(self._parse_features(boundary_feature)())
#         self.structure = structure

#     def execute(self, eopatch):
#         extent_mask = eopatch[self.extent_feature].squeeze(axis=-1)
#         boundary_mask = binary_dilation(
#             extent_mask, selem=self.structure) - extent_mask
#         eopatch[self.boundary_feature] = boundary_mask[..., np.newaxis]

#         return eopatch
def Extent2Boundary(extent_feature, structure):
    """
    Adds boundary mask from extent mask using binary dilation
    """
    extent_mask = extent_feature.squeeze()
    boundary_mask = binary_dilation(extent_mask, selem=structure) * 255.0 - extent_mask
    boundary_feature = boundary_mask[..., np.newaxis] * 255.0
    return boundary_feature


# source path
image_path = '/home/kelin/data/train/'
image_list = glob.glob(image_path + 'change/*.png')
# target path
dir_name = '/home/kelin/data/train/change_edge/'
for i in range(len(image_list)):
    extent_feature = cv2.imread(image_list[i])[..., 0]
    # 将范围转换为边界
    boundary_feature = Extent2Boundary(extent_feature, structure=disk(2))
    basename = os.path.basename(image_list[i])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(dir_name + basename, boundary_feature)
    # break
