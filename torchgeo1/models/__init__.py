# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .farseg import FarSeg, FarSeg_CD, FarSeg_CD_Res, FarSeg_Res

__all__ = ("FarSeg", "FarSeg_CD", "FarSeg_CD_Res", "FarSeg_Res",)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
