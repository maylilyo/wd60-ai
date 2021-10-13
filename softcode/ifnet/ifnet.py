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

    def forward(self, img1, img2, scale_list=[4, 2, 1]):
        # img1, img2: (num_batches, 3, height, width)
        warped_img1 = img1
        warped_img2 = img2

        [batch_size, channel, height, width] = img1.size()

        flow = torch.zeros([batch_size, 4, height, width]).type_as(img1)
        mask = torch.zeros([batch_size, 1, height, width]).type_as(img1)
        # flow: (num_batches, 4, height, width)
        # mask: (num_batches, 1, height, width)

        block = [self.block0, self.block1, self.block2]

        for i in range(3):
            f0, m0 = block[i](
                x=torch.cat((warped_img1, warped_img2, mask), 1),
                flow=flow,
                scale=scale_list[i],
            )
            f1, m1 = block[i](
                x=torch.cat((warped_img2, warped_img1, -mask), 1),
                flow=torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                scale=scale_list[i],
            )

            # flow
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2

            # mask
            mask = mask + (m0 + (-m1)) / 2
            mask = torch.sigmoid(mask)

            # warp
            warped_img1 = warp(img1, flow[:, :2]) * mask
            warped_img2 = warp(img2, flow[:, 2:4]) * (1 - mask)

        return flow
