# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""
import torch
import torch.nn as nn
import math


def auto_pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, auto_pad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c_in, c_out, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c_out * e)
        self.conv1 = Conv(c_in, c_, 1, 1)
        self.conv2 = Conv(c_, c_out, 3, 1, g=g)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    # ch_in, ch_out, number, shortcut, groups, expansion yolov5中没有用到
    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c_out * e)
        self.conv1 = Conv(c_in, c_out, 1, 1)
        self.conv2 = nn.Conv2d(c_in, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = Conv(c_ * 2, c_out)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C3(nn.Module):

    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c_out * e)
        self.conv1 = Conv(c_in, c_, 1, 1)
        self.conv2 = Conv(c_in, c_, 1, 1)
        self.conv3 = Conv(2 * c_in, c_out, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv2(torch.cat((self.conv2(x), self.m(self.conv1), 1)))


class SPPF(nn.Module):

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_ = c_in // 2
        self.conv1 = Conv(c_in, c_, 1, 1)
        self.conv2 = Conv(4 * c_, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), 1))


class Concat(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)
