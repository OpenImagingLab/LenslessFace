# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .global_dw_neck import GlobalDepthWiseNeck,GlobalDepthWiseNeck_hybridL

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
           'GlobalDepthWiseNeck','GlobalDepthWiseNeck_hybridL']
