import torch
from abc import ABCMeta, abstractclassmethod

torch.random.manual_seed(0)
y = torch.arange(10)
x = torch.arange(5)
xv, yv = torch.meshgrid(x, y)
print(xv.shape, yv.shape)
grid = torch.stack((xv, yv), 2)
# grid = torch.cat((xv ,yv),2)
print(grid.shape)