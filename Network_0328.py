import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

CH_FOLD2 = 1


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, target_size=None):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=target_size, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    def __init__(self, entropy_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        # Encoder
        self.Conv1 = conv_block(entropy_ch, 32)
        self.Conv2 = conv_block(32, 64)
        self.Conv3 = conv_block(64, 128)
        self.Conv4 = conv_block(128, 256)
        self.Conv5 = conv_block(256, 512) 

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.Up5 = up_conv(512, 256, (3, 3))
        self.Up_conv5 = conv_block(512, 256)

        self.Up4 = up_conv(256, 128, (7, 7))
        self.Up_conv4 = conv_block(256, 128)

        self.Up3 = up_conv(128, 64, (14, 14))
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32, (28, 28))
        self.Up_conv2 = conv_block(64, 32)

        # Output
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 84))
        self.fc = nn.Linear(84, 84)


        self.bottleneck = None

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)  # [N, 32, 28, 28]
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # [N, 64, 14, 14]
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # [N, 128, 7, 7]
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # [N, 256, 3, 3]
        self.bottleneck = x4  # 存储瓶颈层输出
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # [N, 512, 1, 1]

        # self.bottleneck = x5  # 存储瓶颈层输出

        # Decoder
        d5 = self.Up5(x5)  # [N, 256, 3, 3]
        d5 = torch.cat([x4, d5], dim=1)
        d5 = self.Up_conv5(d5)


        d4 = self.Up4(d5)  # [N, 128, 7, 7]
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.Up_conv4(d4)


        d3 = self.Up3(d4)  # [N, 64, 14, 14]
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.Up_conv3(d3)


        d2 = self.Up2(d3)  # [N, 32, 28, 28]
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.Up_conv2(d2)




        out = self.Conv_1x1(d2)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)


        return out
