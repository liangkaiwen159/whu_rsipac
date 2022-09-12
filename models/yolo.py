# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
import argparse
from ast import arg
import contextlib
from copy import deepcopy
from pathlib import Path
import sys
import platform
import os
import yaml
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *


class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=2, anchors=(), c_out=(), inplace=True):
        super().__init__()
        self.nc = nc  #分类个数
        self.no = nc + 5  #每个anchor的输出
        self.nl = len(anchors)  #检测层的个数
        self.na = len(anchors[0]) // 2  #anchor的个数
        self.grid = [torch.empty(1)] * self.nl
        self.anchor_grid = [torch.empty[1]] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  #shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in c_out)
        self.inplace = inplace

    def forward(self, x):
        z = []  # 推理输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape  #(bs, 3*(n_cls+5), 20, 20) -> (bs, 3, 20, 20, (n_cls+5))
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device


class BaseModel(nn.Module):

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


class DetectionModel(BaseModel):
    # model, input channels, number of classes
    def __init__(self, cfg='yolov5l.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), c_out_list=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)


def parse_model(yaml, c_out_list):
    anchors, nc, gd, gw = yaml['anchors'], yaml['nc'], yaml['depth_multiple'], yaml['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c_out = [], [], c_out_list[-1]
    for i, (f, n, m, args) in enumerate(yaml['backbone'] + yaml['head']):
        m = eval(m) if isinstance(m, str) else m  #从common加载model
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in (Conv, Bottleneck, SPPF, C3):
            c_in, c_out = c_out_list[f], args[0]
            if c_out != no:
                c_out = make_divisible(c_out * gw, 8)
            args = [c_in, c_out, *args[1:]]
            if m in (C3, ):
                args.insert(2, n)
                n = 1  #？？？
        elif m is nn.BatchNorm2d:
            args = [c_out_list[f]]
        elif m is Concat:
            c_out = sum(c_out_list[x] for x in f)
        elif m is Detect:
            args.append([c_out_list[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
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
    cfg = 'models\yolov5l.yaml'
    model = DetectionModel(cfg)
