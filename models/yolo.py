# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
import sys
import platform
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common_kevin import C3, SPPF, Conv, Concat
import argparse
from ast import arg
import contextlib
from copy import deepcopy
from utils.autoanchor import check_anchor_order
import torch.nn as nn
from utils.torch_utils import initialize_weights
import torch
import math
from torchvision.models._utils import IntermediateLayerGetter


class Seg(nn.Module):

    def __init__(self):
        super().__init__()
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv1 = Conv(256, 128, k=1)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv2 = Conv(256, 64, k=1)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv3 = Conv(128, 16, k=1)
        self.conv4 = Conv(16, 2, k=1)

    def forward(self, x):
        input_4 = x[0]
        input_2 = x[1]
        input_0 = x[2]
        out = self.conv1(self.ups1(input_4))
        out = torch.cat((out, input_2), dim=1)
        out = self.conv2(self.ups2(out))
        out = torch.cat((out, input_0), dim=1)
        out = self.conv4(self.conv3(self.ups3(out)))
        return out


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2)**2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        yv, xv = torch.meshgrid([torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):

    def __init__(self, cfg='yolov5l.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.model, self.save = parse_model(deepcopy(self.yaml), [ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-2]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)

    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.model[:-1]:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        for m in [self.model[-1]]:
            seg_out = [y[j] for j in m.f]
            seg_out = m(seg_out)
        return x, seg_out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-2]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s)**2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(yaml, c_out_list):
    anchors, nc, gd, gw = yaml['anchors'], yaml['nc'], yaml['depth_multiple'], yaml['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c_out = [], [], c_out_list[-1]
    for i, (f, n, m, args) in enumerate(yaml['backbone'] + yaml['head']):
        m = eval(m) if isinstance(m, str) else m  # 从common加载model
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in (Conv, SPPF, C3):
            c_in, c_out = c_out_list[f], args[0]
            if c_out != no:
                c_out = make_divisible(c_out * gw, 8)
            args = [c_in, c_out, *args[1:]]
            if m in (C3, ):
                args.insert(2, n)
                n = 1  # ？？？
        elif m is nn.BatchNorm2d:
            args = [c_out_list[f]]
        elif m is Concat:
            c_out = sum(c_out_list[x] for x in f)
        elif m is Detect:
            args.append([c_out_list[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Seg:
            args = []
        else:
            c_out = c_out_list[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            c_out_list = []
        c_out_list.append(c_out)
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


if __name__ == '__main__':
    cfg = 'models/yolov5l.yaml'
    model = Model(cfg)
    # all_m = [m for m in model.children()]
    # for i, sub_m in enumerate(all_m[0]):
    #     sub_m.eval()
    model.eval()
    torch.random.manual_seed(0)
    img = torch.randn((16 if torch.cuda.is_available() else 1, 3, 640, 640))
    dets, seg = model(img)

    if model.training:
        for i, det in enumerate(dets):
            print('*' * 20, 'train', '*' * 20)
            print('det{}:'.format(i + 1), det.shape)
    else:
        print('*' * 20, 'eval', '*' * 20)
        det = dets[0]
        dets = dets[1]
        print('anchors:', det.shape)
        for i, sub_det in enumerate(dets):
            print('det{}:'.format(i + 1), sub_det.shape)
    print('seg:', seg.shape)
