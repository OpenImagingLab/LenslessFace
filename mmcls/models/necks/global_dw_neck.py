# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import NECKS


@NECKS.register_module()
class GlobalDepthWiseNeck(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=128,
                 kernel_size=(7, 6)
                 ):
        super(GlobalDepthWiseNeck, self).__init__()
        self.dw_conv = ConvModule(in_channels, in_channels,
                                  kernel_size=kernel_size, act_cfg=None,
                                  groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # print(x.shape)
        x = self.dw_conv(x)
        x = self.bn(x)
        # print(x.shape)
        assert x.shape[2] == 1 and x.shape[3] == 1
        x = x.flatten(1)
        x = self.fc(x)
        return x


@NECKS.register_module()
class GlobalDepthWiseNeck_hybridL(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=128,
                 kernel_size=(7, 6)
                 ):
        super(GlobalDepthWiseNeck_hybridL, self).__init__()
        self.dw_conv = ConvModule(in_channels, in_channels,
                                  kernel_size=kernel_size, act_cfg=None,
                                  groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x,kernel,x0):
        x = self.dw_conv(x)
        x = self.bn(x)
       # print(x.shape)
        assert x.shape[2] == 1 and x.shape[3] == 1
        x = x.flatten(1)
        x = self.fc(x)
        return x,kernel,x0


