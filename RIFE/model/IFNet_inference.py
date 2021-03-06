import torch
import torch.nn as nn
import torch.nn.functional as F

from model.warplayer import warp
from model.refine import Contextnet, Unet

# remove deconv layer


def conv(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    # Coarse-to-Fine structure
    def __init__(
        self,
        in_planes,
        c=64,
    ):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        # conv, stride=2 x 2
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            # conv, stride=1 x 8
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(
                x,
                scale_factor=1. / scale,
                mode="bilinear",
                align_corners=False,
            )

        if flow is not None:
            flow = F.interpolate(
                flow,
                scale_factor=1. / scale,
                mode="bilinear",
                align_corners=False,
            )
            flow *= 1. / scale
            x = torch.cat((x, flow), 1)

        x = self.conv0(x)
        x = self.convblock(x) + x

        tmp = self.lastconv(x)
        tmp = F.interpolate(
            tmp,
            scale_factor=scale * 2,
            mode="bilinear",
            align_corners=False,
        )

        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]

        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1]):
        # x = [img0(0, 1, 2), img1(3, 4, 5), gt(6)]
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None

        stu = [self.block0, self.block1, self.block2]

        # img0, img1 : (batch_size/2, 3, 224, 224)
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])

            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)

        return merged[2]
