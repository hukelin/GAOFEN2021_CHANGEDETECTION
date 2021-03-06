# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg, ChangeModel
from .farseg import FarSeg
from .fccd import FCEF, FCSiamConc, FCSiamDiff
from .fcn import FCN

__all__ = (
    "ChangeMixin",
    "ChangeStar",
    "ChangeStarFarSeg",
    "FarSeg",
    "FCN",
    "FCEF",
    "FCSiamConc",
    "FCSiamDiff",
    "ChangeModel"
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
