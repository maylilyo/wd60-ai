import torch
import torch.nn as nn
from icecream import ic

from gridnet.modules import LateralBlock, DownSamplingBlock, UpSamplingBlock


class GridNet(nn.Module):
    # Image Synthesis Network
    def __init__(self, num_out_channel=3, grid_channel_list=[32, 64, 96]):
        # num_row = 3, num_column = 6, num_channel_list = [32, 64, 96]):
        super().__init__()

        self.num_row = 3
        self.num_column = 6
        self.num_channel_list = grid_channel_list

        assert len(grid_channel_list) == self.num_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(70, self.num_channel_list[0])

        for r, num_channel in enumerate(self.num_channel_list):
            for c in range(self.num_column - 1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(num_channel, num_channel))

        for r, (input_channel, output_channel) in enumerate(zip(self.num_channel_list[:-1], self.num_channel_list[1:])):
            for c in range(int(self.num_column / 2)):
                # 00, 10일 때 예외처리
                if r == 0 and c == 0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(128, output_channel))
                elif r == 1 and c == 0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(192, output_channel))
                else:
                    setattr(self, f'down_{r}_{c}', DownSamplingBlock(input_channel, output_channel))

        for r, (input_channel, output_channel) in enumerate(zip(self.num_channel_list[1:], self.num_channel_list[:-1])):
            for c in range(int(self.num_column / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(input_channel, output_channel))

        self.lateral_final = LateralBlock(self.num_channel_list[0], num_out_channel)

    def forward(self, x_l1, x_l2, x_l3):
        # x_l1: [num_batches, 70, 256, 448]
        # x_l2: [num_batches, 128, 128, 224]
        # x_l3: [num_batches, 192, 64, 112]

        state_00 = self.lateral_init(x_l1)
        state_10 = self.down_0_0(x_l2)
        state_20 = self.down_1_0(x_l3)

        # 01의 입력을 warp로 바꾼 결과
        # Down Block Stage
        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)
        # state_00, 01, 02 : [num_batches, 32, 256, 448]
        # state_10, 11, 12 : [num_batches, 64, 128, 224]
        # state_20, 21, 22 : [num_batches, 96, 64, 112]

        # Up Block Stage
        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)
        # state_23, 24, 25 : [num_batches, 96, 64, 112]
        # state_13, 14, 15 : [num_batches, 64, 128, 224]
        # state_03, 04, 05 : [num_batches, 32, 256, 448]

        # print(self.lateral_final(state_05).shape)
        return self.lateral_final(state_05)


if __name__ == '__main__':
    W = 448
    H = 328
    N = 2
    x_l1 = torch.rand(size=(N, 70, H, W)).cuda()
    x_l2 = torch.rand(size=(N, 128, H//2, W//2)).cuda()
    x_l3 = torch.rand(size=(N, 192, H//4, W//4)).cuda()
    model = GridNet(num_out_channel=3).cuda()
    res = model(x_l1, x_l2, x_l3)
    print(res.shape)
