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

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)         # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.sigmoid(q)
        return U * q                # 广播机制

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = Double_Conv_BN_ReLU(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

#  标准的Unet的MaxPool+Doubel Conv
class unet_down(nn.Module):
    def __init__(self, in_ch, out_ch, have_scSE=False):
        super(unet_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_Conv_BN_ReLU(in_ch, out_ch)
        )
        self.have_scSE=have_scSE
        if self.have_scSE:
            self.scSE = scSE(in_ch)


    def forward(self, x):
        if self.have_scSE:
            x=self.scSE(x)
        x = self.mpconv(x)
        return x

#  标准的Unet的Upsample或ConvTranspose2d+Doubel Conv
class unet_up(nn.Module):
    def __init__(self, in_ch, out_ch, mode='ConvTranspose2d', have_scSE=False):
        """
        :param in_ch:
        :param out_ch:
        :param mode:Upsample or ConvTranspose2d
        """
        super(unet_up, self).__init__()
        if mode=='ConvTranspose2d':
            self.up = nn.ConvTranspose2d(in_ch-out_ch, in_ch-out_ch, kernel_size=3, stride=2,padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.conv = Double_Conv_BN_ReLU(in_ch, out_ch)
        self.have_scSE = have_scSE
        if self.have_scSE:
            self.scSE = scSE(in_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size()[2] < x2.size()[2]:
            x1 = F.pad(x1, (0,0,0,x2.size()[2]-x1.size()[2]))
        if x1.size()[3] < x2.size()[3]:
            x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3],0,0))
        if x1.size()[2] > x2.size()[2]:
            x2 = F.pad(x2, (0,0,0,x1.size()[2]-x2.size()[2]))
        if x1.size()[3] > x2.size()[3]:
            x2 = F.pad(x2, (0, x1.size()[3] - x2.size()[3],0,0))
        x = torch.cat([x2, x1], dim=1)
        if self.have_scSE:
            x=self.scSE(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x

# 卷积（311）+ ReLU
class Conv_ReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_ReLU, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 卷积（311）+ ReLU+MaxPooling
class Conv_ReLU_MaxPool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_ReLU_MaxPool, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 卷积（311）+BN+ ReLU
class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_BN_ReLU, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 卷积（311）+BN+ ReLU+MaxPooling
class Conv_BN_ReLU_MaxPool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_BN_ReLU_MaxPool, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 两组 卷积（311）+ BN + ReLU
class Double_Conv_ReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_Conv_ReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 两组 卷积（311）+ BN + ReLU
class Double_Conv_BN_ReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_Conv_BN_ReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTransposeUnetscSE(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,have_sigmoid=True):
        super(ConvTransposeUnetscSE, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.down1 = unet_down(32, 64,have_scSE=True)
        self.down2 = unet_down(64, 128,have_scSE=True)
        self.down3 = unet_down(128, 256,have_scSE=True)
        self.down4 = unet_down(256, 512,have_scSE=True)

        self.up4 = unet_up(512+256, 256, mode='ConvTranspose2d', have_scSE=True)
        self.up3 = unet_up(256+128, 128, mode='ConvTranspose2d', have_scSE=True)
        self.up2 = unet_up(128+64, 64, mode='ConvTranspose2d', have_scSE=True)
        self.up1 = unet_up(64+32, 32, mode='ConvTranspose2d', have_scSE=True)
        self.outc = outconv(32, out_channels)
        self.have_sigmoid=have_sigmoid

    def forward(self, x):
        in0 = self.inc(x)
        d1 = self.down1(in0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4, d3)
        u3 = self.up3(u4, d2)
        u2 = self.up2(u3, d1)
        u1 = self.up1(u2, in0)
        u0 = self.outc(u1)
        if self.have_sigmoid:
            out = torch.sigmoid(u0)
        else:
            out=u0
        return out


if __name__=='__main__':
    model=ConvTransposeUnetscSE(3,1)
    print(model)
    data=torch.ones((1,3,584,565))
    out=model(data)
    print(out.shape)