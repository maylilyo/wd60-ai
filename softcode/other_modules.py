import torch
import torch.nn as nn
from icecream import ic


class context_extractor_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        # ic.disable()
        ic(x.shape)

        layer1 = self.layer1(x)
        ic(layer1.shape)

        layer2 = self.layer2(layer1)
        ic(layer2.shape)

        layer3 = self.layer3(layer2)
        ic(layer3.shape)

        return [layer1, layer2, layer3]


class Matric_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        class Decoder(nn.Module):
            def __init__(self, l_num):
                super().__init__()

                self.conv_relu = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=l_num*32*2,
                        out_channels=l_num*32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=l_num*32,
                        out_channels=l_num*32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )

            def forward(self, x1, x2):

                x1 = torch.cat((x1, x2), dim=1)
                x1 = self.conv_relu(x1)
                return x1

        # Color를 유지하면서 3 channels를 12 channels로 변경
        self.conv_img = nn.Conv2d(
            in_channels=3,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # 양쪽 사진의 loss를 1 channel에서 4 channels로 바꿈
        # 동시에 background의 중요도를 계산
        self.conv_metric = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_l1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.down_l2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.down_l3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )
        self.up_l3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )
        self.up_l2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )
        self.up_l1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.out_seq = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # 중간 단계가 아니라 output 직전에 ReLU를 쓰는 게 이상함
            # nn.ReLU(inplace=False),
        )
        self.Decoder3 = Decoder(3)
        self.Decoder2 = Decoder(2)
        self.Decoder1 = Decoder(1)

    def forward(self, tenMetric, img1):
        conv_img = self.conv_img(img1)
        conv_metric = self.conv_metric(tenMetric)
        ten_input_l0 = torch.cat([conv_metric, conv_img], dim=1)
        ten_d_l1 = self.down_l1(ten_input_l0)
        ten_d_l2 = self.down_l2(ten_d_l1)
        ten_d_l3 = self.down_l3(ten_d_l2)
        ten_middle = self.middle(ten_d_l3)

        ten_u_l3 = self.Decoder3(ten_d_l3, ten_middle)
        ten_u_l2 = self.up_l3(ten_u_l3)

        ten_u_l2 = self.Decoder2(ten_d_l2, ten_u_l2)
        ten_u_l1 = self.up_l2(ten_u_l2)

        ten_u_l1 = self.Decoder1(ten_d_l1, ten_u_l1)
        ten_out = self.up_l1(ten_u_l1)

        return self.out_seq(ten_out)


if __name__ == '__main__':
    W = 448
    H = 328
    test_tenmatric = torch.rand(size=(2, 1, H, W))
    test_img = torch.rand(size=(2, 3, H, W))
    model = Matric_UNet()
    res = model(test_tenmatric, test_img)
    print(model)
    print(res.shape)
