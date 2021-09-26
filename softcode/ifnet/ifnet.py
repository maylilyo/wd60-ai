import torch
import torch.nn as nn

from .ifblock import IFBlock
from .warp import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)

    def forward(self, x, scale_list=[4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]

        for i in range(3):
            f0, m0 = block[i](
                x=torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1),
                flow=flow,
                scale=scale_list[i],
            )
            f1, m1 = block[i](
                x=torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                flow=torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                scale=scale_list[i],
            )

            # flow
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2

            # mask
            mask = mask + (m0 + (-m1)) / 2
            mask = torch.sigmoid(mask)

            # warp
            warped_img0 = warp(img0, flow[:, :2]) * mask
            warped_img1 = warp(img1, flow[:, 2:4]) * (1 - mask)

        return flow
