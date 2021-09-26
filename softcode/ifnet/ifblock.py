import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
