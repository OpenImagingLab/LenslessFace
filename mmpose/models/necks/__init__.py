# Copyright (c) OpenMMLab. All rights reserved.
from .fpn import FPN
from .gap_neck import GlobalAveragePooling,GlobalAveragePooling_hybridLoss
from .posewarper_neck import PoseWarperNeck
from .tcformer_mta_neck import MTA

__all__ = ['GlobalAveragePooling','GlobalAveragePooling_hybridLoss','PoseWarperNeck', 'FPN', 'MTA']
