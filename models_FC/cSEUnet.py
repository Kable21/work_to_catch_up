# -*- coding: utf-8 -*-
import os
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import os


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)             # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)        # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)     # shape: [bs, c, 1, 1]
        z = self.sigmoid(z)
        return U * z.expand_as(U)



