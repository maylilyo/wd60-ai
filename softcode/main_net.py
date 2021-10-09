# import cv2
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, l1_loss
from icecream import ic

from .gridnet.net_module import GridNet
# from pwc.utils.flow_utils import show_compare
from .pwc.pwc_network import Network as PWCNet
from .ifnet.ifnet import IFNet

from .softsplatting.softsplat import _softspalt
from .softsplatting.run import backwarp
from .other_modules import context_extractor_layer, Matric_UNet


class Main_net(nn.Module):
    def __init__(self, model_option):
        super().__init__()

        self.tag = 'pwcnet'  # pwcnet, ifnet
        self.shape = model_option['shape']  # [height, width]
        self.feature_extractor_1 = context_extractor_layer()
        self.feature_extractor_2 = context_extractor_layer()
        self.beta1 = nn.Parameter(torch.Tensor([-1]))
        self.beta2 = nn.Parameter(torch.Tensor([-1]))
        self.Matric_UNet = Matric_UNet()
        self.grid_net = GridNet()

        if self.tag == 'pwcnet':
            self.flow_extractor1to2 = PWCNet()
            self.flow_extractor2to1 = PWCNet()
        elif self.tag == 'ifnet':
            self.flow_extractor = IFNet()

    def scale_flow_zero(self, flow):
        SCALE = 20.0
        intHeight, intWidth = self.shape

        raw_scaled = (SCALE / 1) * interpolate(
            input=flow,
            size=(intHeight, intWidth),
            mode='bilinear',
            align_corners=False,
        )
        return raw_scaled

    def scale_flow(self, flow):
        # https://github.com/sniklaus/softmax-splatting/issues/12

        SCALE = 20.0
        intHeight, intWidth = self.shape

        raw_scaled = (SCALE / 1) * interpolate(
            input=flow,
            size=(intHeight, intWidth),
            mode='bilinear',
            align_corners=False,
        )
        half_scaled = (SCALE / 2) * interpolate(
            input=flow,
            size=(intHeight // 2, intWidth // 2),
            mode='bilinear',
            align_corners=False,
        )
        quarter_scaled = (SCALE / 4) * flow

        return [raw_scaled, half_scaled, quarter_scaled]

    def scale_tenMetric(self, tenMetric):
        intHeight, intWidth = self.shape

        raw_scaled = tenMetric
        half_scaled = interpolate(
            input=tenMetric,
            size=(intHeight // 2, intWidth // 2),
            mode='bilinear',
            align_corners=False,
        )
        quarter_scaled = interpolate(
            input=tenMetric,
            size=(intHeight // 4, intWidth // 4),
            mode='bilinear',
            align_corners=False,
        )
        return [raw_scaled, half_scaled, quarter_scaled]

    def forward(self, img1, img2):
        ic.disable()

        ic(img1.shape)
        # img1, img2: (num_batches, 3, height, width)

        # ↓ Optional Information Provider

        feature_pyramid1 = self.feature_extractor_1(img1)
        feature_pyramid2 = self.feature_extractor_2(img2)

        ic(len(feature_pyramid1))
        ic(feature_pyramid1[0].shape)
        ic(feature_pyramid1[1].shape)
        ic(feature_pyramid1[2].shape)

        # feature_pyramid1, feature_pyramid2: [layer1, layer2, layer3]
        # layer1: (num_batches, 32, height, width)
        # layer2: (num_batches, 64, height / 2, width / 2)
        # layer3: (num_batches, 96, height / 4, width / 4)

        # ↓ Optical Flow Estimator

        flow_1to2 = self.flow_extractor1to2(img1, img2)
        flow_2to1 = self.flow_extractor1to2(img2, img1)
        # flow_1to2, flow_2to1: (num_batches, 2, height / 4, width / 4)

        flow_1to2_zero = self.scale_flow_zero(flow_1to2)
        flow_2to1_zero = self.scale_flow_zero(flow_2to1)

        if self.tag == 'pwcnet':
            flow_1tot = flow_1to2 * 0.5
            flow_2tot = flow_2to1 * 0.5

        elif self.tag == 'ifnet':
            flow_all = self.flow_extractor(img1, img2)
            channel = flow_all.shape[1] // 2
            flow_1tot = flow_all[:, :channel]
            flow_2tot = flow_all[:, channel:]

        flow_1to2_pyramid = self.scale_flow(flow_1tot)
        flow_2to1_pyramid = self.scale_flow(flow_2tot)

        target_1to2 = backwarp(img2, flow_1to2_zero)
        target_2to1 = backwarp(img1, flow_2to1_zero)

        # TEST
        # show_compare(flow_1to2_pyramid[0].squeeze().cpu().detach().numpy().transpose(1,2,0),
        # flow_1to2_pyramid[0].squeeze().cpu().detach().numpy().transpose(1,2,0))

        ic(len(flow_1to2_pyramid))
        ic(flow_1to2_pyramid[0].shape)
        ic(flow_1to2_pyramid[1].shape)
        ic(flow_1to2_pyramid[2].shape)

        # flow_1to2_pyramid, flow_2to1_pyramid: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 2, height, width)
        # half_scaled: (num_batches, 2, height / 2, width / 2)
        # quarter_scaled: (num_batches, 2, height / 4, width / 4)

        # ↓ Softmax Splatting

        tenMetric_1to2 = l1_loss(
            input=img1,
            target=target_1to2,
            reduction='none',
        )
        ic(tenMetric_1to2.shape)
        # tenMetric_1to2: (num_batches, 3, height, width)

        tenMetric_1to2 = tenMetric_1to2.mean(1, True)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_1to2 = self.Matric_UNet(tenMetric_1to2, img1)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)

        ic(len(tenMetric_ls_1to2))
        ic(tenMetric_ls_1to2[0].shape)
        ic(tenMetric_ls_1to2[1].shape)
        ic(tenMetric_ls_1to2[2].shape)

        # tenMetric_ls_1to2: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 1, height, width)
        # half_scaled: (num_batches, 1, height / 2, width / 2)
        # quarter_scaled: (num_batches, 1, height / 4, width / 4)

        ic(self.beta1.shape)

        # for i in range(3):
        #     print(nn.MSELoss()(tmp_pyramid[i], flow_1to2_pyramid[i] * 0.5))
        # beta1: (1,)

        warped_img1 = _softspalt(
            tenInput=img1,
            tenFlow=flow_1to2_pyramid[0],
            tenMetric=self.beta1 * tenMetric_ls_1to2[0],
            _type='softmax',
        )
        warped_pyramid1_1 = _softspalt(
            tenInput=feature_pyramid1[0],
            tenFlow=flow_1to2_pyramid[0],
            tenMetric=self.beta1 * tenMetric_ls_1to2[0],
            _type='softmax',
        )
        warped_pyramid1_2 = _softspalt(
            tenInput=feature_pyramid1[1],
            tenFlow=flow_1to2_pyramid[1],
            tenMetric=self.beta1 * tenMetric_ls_1to2[1],
            _type='softmax',
        )
        warped_pyramid1_3 = _softspalt(
            tenInput=feature_pyramid1[2],
            tenFlow=flow_1to2_pyramid[2],
            tenMetric=self.beta1 * tenMetric_ls_1to2[2],
            _type='softmax',
        )

        ic(warped_img1.shape)
        ic(warped_pyramid1_1.shape)
        ic(warped_pyramid1_2.shape)
        ic(warped_pyramid1_3.shape)

        # warped_img1: (num_batches, 3, height, width)
        # warped_pyramid1_1: (num_batches, 32, height, width)
        # warped_pyramid1_2: (num_batches, 64, height / 2, width / 2)
        # warped_pyramid1_3: (num_batches, 96, height / 4, width / 4)

        tenMetric_2to1 = l1_loss(
            input=img2,
            target=target_2to1,
            reduction='none',
        )
        tenMetric_2to1 = tenMetric_2to1.mean(1, True)
        tenMetric_2to1 = self.Matric_UNet(tenMetric_2to1, img2)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)

        warped_img2 = _softspalt(
            tenInput=img2,
            tenFlow=flow_2to1_pyramid[0],
            tenMetric=self.beta2 * tenMetric_ls_2to1[0],
            _type='softmax',
        )
        warped_pyramid2_1 = _softspalt(
            tenInput=feature_pyramid2[0],
            tenFlow=flow_2to1_pyramid[0],
            tenMetric=self.beta2 * tenMetric_ls_2to1[0],
            _type='softmax',
        )
        warped_pyramid2_2 = _softspalt(
            tenInput=feature_pyramid2[1],
            tenFlow=flow_2to1_pyramid[1],
            tenMetric=self.beta2 * tenMetric_ls_2to1[1],
            _type='softmax',
        )
        warped_pyramid2_3 = _softspalt(
            tenInput=feature_pyramid2[2],
            tenFlow=flow_2to1_pyramid[2],
            tenMetric=self.beta2 * tenMetric_ls_2to1[2],
            _type='softmax',
        )

        # warped_img2: (num_batches, 3, height, width)
        # warped_pyramid2_1: (num_batches, 32, height, width)
        # warped_pyramid2_2: (num_batches, 64, height / 2, width / 2)
        # warped_pyramid2_3: (num_batches, 96, height / 4, width / 4)

        # ↓ Image Synthesis Network

        grid_input_l1 = torch.cat(
            [warped_img1, warped_pyramid1_1, warped_img2, warped_pyramid2_1],
            dim=1,
        )
        grid_input_l2 = torch.cat(
            [warped_pyramid1_2, warped_pyramid2_2],
            dim=1,
        )
        grid_input_l3 = torch.cat(
            [warped_pyramid1_3, warped_pyramid2_3],
            dim=1,
        )

        ic(grid_input_l1.shape)
        ic(grid_input_l2.shape)
        ic(grid_input_l3.shape)

        # grid_input_l1: (num_batches, 70, height, width)
        # grid_input_l2: (num_batches, 128, height / 2, width / 2)
        # grid_input_l3: (num_batches, 192, height / 4, width / 4)

        out = self.grid_net(grid_input_l1, grid_input_l2, grid_input_l3)
        ic(out.shape)
        # out: (num_batches, 3, height, width)

        return out


if __name__ == '__main__':
    W = 448
    H = 256
    N = 1

    tenFirst = torch.rand(size=(N, 3, H, W)).cuda()
    tenSecond = torch.rand(size=(N, 3, H, W)).cuda()

    model = Main_net(tenFirst.shape).cuda()

    res = model(tenFirst, tenSecond)
    ic(res.shape)
