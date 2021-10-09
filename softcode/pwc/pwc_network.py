import os
import sys
import math
import getopt

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.constv import ConstV
from .correlation import correlation

# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

# make sure to use cudnn for computational performance
torch.backends.cudnn.enabled = True


arguments_strModel = 'default'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '':
        arguments_strModel = strArgument  # which model to use
    if strOption == '--first' and strArgument != '':
        arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '':
        arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '':
        arguments_strOut = strArgument  # path to where the output should be stored


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.netOne = self.get_new_CNN(3, 16)
        self.netTwo = self.get_new_CNN(16, 32)
        self.netThr = self.get_new_CNN(32, 64)
        self.netFou = self.get_new_CNN(64, 96)
        self.netFiv = self.get_new_CNN(96, 128)
        self.netSix = self.get_new_CNN(128, 196)

    def get_new_CNN(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            )
        )

    def forward(self, input_tensor):
        tenOne = self.netOne(input_tensor)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


class Decoder(nn.Module):
    tmp_list = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None]

    grid_cache = {}
    partial_cache = {}

    def __init__(self, intLevel):
        super().__init__()

        intPrevious = self.tmp_list[intLevel + 1]
        intCurrent = self.tmp_list[intLevel]

        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        if intLevel < 6:
            self.netUpfeat = nn.ConvTranspose2d(
                in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        if intLevel < 6:
            self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        self.netOne = self.get_new_CNN(intCurrent, 128)
        self.netTwo = self.get_new_CNN(intCurrent + 128, 128)
        self.netThr = self.get_new_CNN(intCurrent + 128 + 128, 96)
        self.netFou = self.get_new_CNN(intCurrent + 128 + 128 + 96, 64)
        self.netFiv = self.get_new_CNN(intCurrent + 128 + 128 + 96 + 64, 32)
        self.netSix = nn.Sequential(
            nn.Conv2d(
                in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def get_new_CNN(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            )
        )

    def backwarp(self, input_tensor, flow_tensor):
        key = str(flow_tensor.shape)

        if key in Decoder.grid_cache:
            backwarped_tensor = Decoder.grid_cache[key]
        else:
            horizontal_tensor = torch.linspace(
                start=-1.0 + (1.0 / flow_tensor.shape[3]),
                end=1.0 - (1.0 / flow_tensor.shape[3]),
                steps=flow_tensor.shape[3],
            )
            horizontal_tensor = horizontal_tensor.view(1, 1, 1, -1)
            horizontal_tensor = horizontal_tensor.expand(-1, -1, flow_tensor.shape[2], -1)

            vertical_tensor = torch.linspace(
                start=-1.0 + (1.0 / flow_tensor.shape[2]),
                end=1.0 - (1.0 / flow_tensor.shape[2]),
                steps=flow_tensor.shape[2],
            )
            vertical_tensor = vertical_tensor.view(1, 1, -1, 1)
            vertical_tensor = vertical_tensor.expand(-1, -1, -1, flow_tensor.shape[3])

            backwarped_tensor = torch.cat(
                tensors=[horizontal_tensor, vertical_tensor],
                dim=1,
            )
            # backwarped_tensor = backwarped_tensor.cuda()
            backwarped_tensor = backwarped_tensor.type_as(flow_tensor)

            # # Save grid to cache
            # Decoder.grid_cache[key] = backwarped_tensor

        # if key in Decoder.partial_cache:
        #     partial_tensor = Decoder.partial_cache[key]
        # else:
        #     partial_tensor = flow_tensor.new_ones([flow_tensor.shape[0], 1, flow_tensor.shape[2], flow_tensor.shape[3]])
        #     partial_tensor = partial_tensor.type_as(input_tensor)

        #     # Save partial to cache
        #     Decoder.partial_cache[key] = partial_tensor

        partial_tensor = flow_tensor.new_ones([flow_tensor.shape[0], 1, flow_tensor.shape[2], flow_tensor.shape[3]])
        partial_tensor = partial_tensor.type_as(flow_tensor)

        flow_tensor = torch.cat(
            tensors=[
                flow_tensor[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                flow_tensor[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)
            ],
            dim=1,
        )

        input_tensor = torch.cat([input_tensor, partial_tensor], 1)

        grid = backwarped_tensor + flow_tensor
        grid = grid.permute(0, 2, 3, 1)

        output_tensor = F.grid_sample(
            input=input_tensor,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        mask_tensor = output_tensor[:, -1:, :, :]
        mask_tensor[mask_tensor > 0.999] = 1.0
        mask_tensor[mask_tensor < 1.0] = 0.0

        return output_tensor[:, :-1, :, :] * mask_tensor

    def forward(self, first_tensor, second_tensor, prev_object):
        flow_tensor = None
        feature_tensor = None

        if prev_object is None:
            flow_tensor = None
            feature_tensor = None

            _input = correlation.FunctionCorrelation(
                first_tensor=first_tensor,
                second_tensor=second_tensor,
            )
            tenVolume = F.leaky_relu(
                input=_input,
                negative_slope=0.1,
                inplace=False,
            )

            feature_tensor = torch.cat(
                tensors=[tenVolume],
                dim=1,
            )
        else:
            flow_tensor = self.netUpflow(prev_object['flow_tensor'])
            feature_tensor = self.netUpfeat(prev_object['feature_tensor'])

            second_tensor = self.backwarp(
                input_tensor=second_tensor,
                flow_tensor=flow_tensor * self.fltBackwarp,
            )
            _input = correlation.FunctionCorrelation(
                first_tensor=first_tensor,
                second_tensor=second_tensor,
            )
            tenVolume = F.leaky_relu(
                input=_input,
                negative_slope=0.1,
                inplace=False,
            )
            # corr 对应  tenVolume first_tensor对应x1  flow 对应tenflow
            feature_tensor = torch.cat(
                tensors=[tenVolume, first_tensor, flow_tensor, feature_tensor],
                dim=1,
            )

        feature_tensor = torch.cat([self.netOne(feature_tensor), feature_tensor], 1)
        feature_tensor = torch.cat([self.netTwo(feature_tensor), feature_tensor], 1)
        feature_tensor = torch.cat([self.netThr(feature_tensor), feature_tensor], 1)
        feature_tensor = torch.cat([self.netFou(feature_tensor), feature_tensor], 1)
        feature_tensor = torch.cat([self.netFiv(feature_tensor), feature_tensor], 1)

        flow_tensor = self.netSix(feature_tensor)

        return {
            'flow_tensor': flow_tensor,
            'feature_tensor': feature_tensor
        }


class Refiner(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.refiner = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=16,
                dilation=16,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            )
        )

    def forward(self, input_tensor):
        return self.refiner(input_tensor)


class Network(nn.Module):  # flownet
    def __init__(self):
        super().__init__()

        # 初始化 context refiner
        # self.context_refiners = []
        # for l, ch in enumerate(ConstV.lv_chs):
        # 	layer = Refiner(ch).to(ConstV.my_device)
        # 	# self.add_module(f'ContextNetwork(Lv{l})', layer)
        # 	self.context_refiners.append(layer)

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        # self.netRefiner = Refiner(565)
        self.refiner = Refiner(565)

        # 一定用高斯初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)
        # print('http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch')
        # 不清楚为啥 不能load 似乎只会起反作用，应该是估计的光流没有起到很好的效果？？？
        # 看来输入的光流必须除以255否则不顶用 这根预训练模型绝对有关系

        self.load_state_dict(torch.load('weights/pwc.pt'))

    def forward(self, first_tensor, second_tensor):
        first_tensor = self.netExtractor(first_tensor)
        second_tensor = self.netExtractor(second_tensor)
        if ConstV.mutil_l:
            objEstimate = self.netSix(first_tensor[-1], second_tensor[-1], None)
            # objEstimate['tneFlow'] = [2,2,6,7]

            flow_l1 = objEstimate['flow_tensor']

            objEstimate = self.netFiv(first_tensor[-2], second_tensor[-2], objEstimate)
            # objEstimate['tneFlow'] = [2,2,12,14] [2,529,6,7]

            flow_l2 = objEstimate['flow_tensor']
            # flow_l2 = objEstimate['flow_tensor'] + self.context_refiners[1](objEstimate['feature_tensor'])
            objEstimate = self.netFou(first_tensor[-3], second_tensor[-3], objEstimate)

            # objEstimate['tneFlow'] = [2,2,24,28] [2, 661, 12, 14]
            flow_l3 = objEstimate['flow_tensor']
            # flow_l3 = objEstimate['flow_tensor'] + self.context_refiners[2](objEstimate['feature_tensor'])
            objEstimate = self.netThr(first_tensor[-4], second_tensor[-4], objEstimate)

            # objEstimate['tneFlow'] = [2,2,48,56]
            flow_l4 = objEstimate['flow_tensor']
            # flow_l4 = objEstimate['flow_tensor'] + self.context_refiners[3](objEstimate['feature_tensor'])
            objEstimate = self.netTwo(first_tensor[-5], second_tensor[-5], objEstimate)
        # shapels.append(objEstimate['feature_tensor'].shape[1])
        # print(shapels)
            flow_l5 = objEstimate['flow_tensor'] + self.refiner(objEstimate['feature_tensor'])
        # objEstimate['tneFlow'] = [2,2,96,112]
        # 貌似是把最后一层的尺寸强行拉到 原始输入尺寸 即扩大四倍 然后数值也乘4
        # if l == args.output_level:
        #                 flow = F.upsample(flow, scale_factor = 2 ** (args.num_levels - args.output_level - 1), mode = 'bilinear') * 2 ** (args.num_levels - args.output_level - 1)
        #                 flows.append(flow)

        # flow_l5 = F.upsample(flow_l5, scale_factor=2 ** (2), mode='bilinear')
            if ConstV.integrated:
                return flow_l5
            if ConstV.train_or_test == 'train':
                # 后面会× 20的，不用再乘了
                flow_l5 = F.upsample(
                    flow_l5, scale_factor=2 ** (2), mode='bilinear')
                return [flow_l1, flow_l2, flow_l3, flow_l4, flow_l5]
            else:
                return flow_l5
        else:
            # shapels = []
            objEstimate = self.netSix(first_tensor[-1], second_tensor[-1], None)
            # objEstimate['tneFlow'] = [2,2,6,7]

            objEstimate = self.netFiv(first_tensor[-2], second_tensor[-2], objEstimate)
            # objEstimate['tneFlow'] = [2,2,12,14] [2,529,6,7]

            objEstimate = self.netFou(first_tensor[-3], second_tensor[-3], objEstimate)
            # shapels.append(objEstimate['feature_tensor'].shape[1])

            objEstimate = self.netThr(first_tensor[-4], second_tensor[-4], objEstimate)

            # flow_l4 = objEstimate['flow_tensor'] + self.context_refiners[3](objEstimate['feature_tensor'])
            objEstimate = self.netTwo(first_tensor[-5], second_tensor[-5], objEstimate)

            # flow_coarse + flow_fine
            return objEstimate['flow_tensor'] + self.refiner(objEstimate['feature_tensor'])


# netNetwork = None
#
#
#
# def estimate(first_tensor, second_tensor):
# 	global netNetwork
#
# 	if netNetwork is None:
# 		netNetwork = Network().cuda().eval()
#
#
# 	assert(first_tensor.shape[1] == second_tensor.shape[1])
# 	assert(first_tensor.shape[2] == second_tensor.shape[2])
#
# 	intWidth = first_tensor.shape[2]
# 	intHeight = first_tensor.shape[1]
#
# 	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
# 	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#
# 	tenPreprocessedFirst = first_tensor.cuda().view(1, 3, intHeight, intWidth)
# 	tenPreprocessedSecond = second_tensor.cuda().view(1, 3, intHeight, intWidth)
#
# 	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
# 	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
#
# 	tenPreprocessedFirst = F.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
# 	tenPreprocessedSecond = F.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#
# 	flow_tensor = 20.0 * F.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
#
# 	flow_tensor[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
# 	flow_tensor[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
#
# 	return flow_tensor[0, :, :, :].cpu()
#
#
#
#
# if __name__ == '__main__':
# 	first_tensor = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
# 	second_tensor = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
# 	#(3, 436, 1024)
# 	output_tensor = estimate(first_tensor, second_tensor)
#
# 	objOutput = open(arguments_strOut, 'wb')
#
# 	numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
# 	numpy.array([ output_tensor.shape[2], output_tensor.shape[1] ], numpy.int32).tofile(objOutput)
# 	numpy.array(output_tensor.detach().numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
#
# 	objOutput.close()
#
