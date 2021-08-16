import cv2
import numpy
import torch
import torch.nn.functional as F

from softsplatting.softsplat import _softspalt


backwarp_cache = {}


def backwarp(input_tensor, flow_tensor):
    key = flow_tensor.shape
    if key not in backwarp_cache:
        horizontal_tensor = torch.linspace(
            start=-1.0 + (1.0 / flow_tensor.shape[3]),
            end=1.0 - (1.0 / flow_tensor.shape[3]),
            steps=flow_tensor.shape[3],
        )
        horizontal_tensor = horizontal_tensor.view(1, 1, 1, -1)
        horizontal_tensor = horizontal_tensor.expand(-1, -1, flow_tensor.shape[2], -1,)

        vertical_tensor = torch.linspace(
            start=-1.0 + (1.0 / flow_tensor.shape[2]),
            end=1.0 - (1.0 / flow_tensor.shape[2]),
            steps=flow_tensor.shape[2],
        )
        vertical_tensor = vertical_tensor.view(1, 1, -1, 1)
        vertical_tensor = vertical_tensor.expand(-1, -1, -1, flow_tensor.shape[3])

        backwarped_tensor = torch.cat([horizontal_tensor, vertical_tensor], 1)
        backwarped_tensor = backwarped_tensor.cuda()

        backwarp_cache[key] = backwarped_tensor
    else:
        backwarped_tensor = backwarp_cache[key]

    flow_tensor = torch.cat(
        tensors=[
            flow_tensor[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
            flow_tensor[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)
        ],
        dim=1)

    grid = backwarped_tensor + flow_tensor
    grid = grid.permute(0, 2, 3, 1)

    return F.grid_sample(
        input=input_tensor,
        grid=grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )


if __name__ == '__main__':
    W = 448
    H = 328
    N = 2
    first_tensor = torch.rand(size=(N, 32, H, W)).cuda()
    flow_tensor = torch.rand(size=(N, 2, H, W)).cuda()
    tenMetric = torch.rand(size=(N, 1, H, W)).cuda()
    tenSoftmax = _softspalt(
        input_tensor=first_tensor,
        flow_tensor=flow_tensor * 0.5,
        tenMetric=-20.0 * tenMetric,
        strType='softmax',
    )
