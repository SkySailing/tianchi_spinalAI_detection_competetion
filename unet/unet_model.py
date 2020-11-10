""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

from unet.unet_parts import *
from torchsummary import summary


class UNet_twoPart(nn.Module):
    def __init__(self, n_channels, n_classes=(6,5), bilinear=True):
        super(UNet_twoPart, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_part1 = OutConv(64, self.n_classes[0])
        self.outc_part2 = OutConv(64, self.n_classes[1])


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xpart1 = self.up1(x5, x4)
        xpart1 = self.up2(xpart1, x3)
        xpart1 = self.up3(xpart1, x2)
        xpart1 = self.up4(xpart1, x1)
        logits_part1 = self.outc_part1(xpart1)
        xpart2 = self.up1(x5, x4)
        xpart2 = self.up2(xpart2, x3)
        xpart2 = self.up3(xpart2, x2)
        xpart2 = self.up4(xpart2, x1)
        logits_part2 = self.outc_part2(xpart2)
        logits = torch.cat([logits_part1,logits_part2],1)
        idx = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5]
        logits = logits[:,idx,:,:]
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_double(nn.Module):
    def __init__(self):
        super(UNet_double, self).__init__()
        self.disc_Unet = UNet(1, 6)
        self.vert_Unet = UNet(1, 5)
        self.idx_swap = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5]

    def forward(self, x):
        disc_Unet_output = self.disc_Unet(x)
        vert_Unet_output = self.vert_Unet(x)
        logits = torch.cat([disc_Unet_output,vert_Unet_output],1)
        logits = logits[:, self.idx_swap, :, :]
        return logits

if __name__ == '__main__':
    unet = UNet(1, 11)
    summary(unet, (1, 256, 256), device='cpu')