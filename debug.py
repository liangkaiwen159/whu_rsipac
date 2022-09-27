import torch
from abc import ABCMeta, abstractclassmethod
import numpy as np
# torch.random.manual_seed(0)
# y = torch.arange(10)
# x = torch.arange(5)
# xv, yv = torch.meshgrid(x, y)
# print(xv.shape, yv.shape)
# grid = torch.stack((xv, yv), 2)
# # grid = torch.cat((xv ,yv),2)
# print(grid.shape)

# r = torch.tensor([[[8.20000, 7.92308], [17.00000, 8.53846]], [[5.12500, 3.43333], [10.62500, 3.70000]],
#                   [[2.48485, 4.47826], [5.15152, 4.82609]]])

# j = torch.max(r, 1. / r)
# print(j)
# j = j.max(2)
# print(j)
# j = j[0]
# print(j)

a = torch.tensor([[3, 4, 5, 6, 7, 8]])
t1 = 0
b = torch.tensor([[7, 8, 9, 10, 11, 12]])
t2 = 8
# x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
x = torch.where((a > t1) & (b == t2))
print(x)
print(a[0, 1])
