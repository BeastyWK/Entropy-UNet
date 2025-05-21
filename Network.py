import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

CH_FOLD2 = 1

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out, target_size=None):
        super(up_conv, self).__init__()
        self.target_size = target_size
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2 if target_size is None else None, size=target_size, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self,entropy_ch=3,output_ch=1):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=entropy_ch,ch_out=int(32*CH_FOLD2))
        self.Conv2 = conv_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))

        self.Up5 = up_conv(ch_in=int(512*CH_FOLD2),ch_out=int(256*CH_FOLD2), target_size=(3, 3))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = up_conv(ch_in=int(256*CH_FOLD2),ch_out=int(128*CH_FOLD2), target_size=(7, 7))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))

        self.Up3 = up_conv(ch_in=int(128*CH_FOLD2),ch_out=int(64*CH_FOLD2), target_size=(14, 14))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))

        self.Up2 = up_conv(ch_in=int(64*CH_FOLD2),ch_out=int(32*CH_FOLD2), target_size=(28, 28))
        self.Up_conv2 = conv_block(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))


        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2),output_ch,kernel_size=1,stride=1,padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 84))
        self.fc = nn.Linear(84, 84) 



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)


        d1 = self.Conv_1x1(d2)
        out = self.avgpool(d1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
