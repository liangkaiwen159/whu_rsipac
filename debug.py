import torch
import torch.nn as nn


class test_init(nn.Module):

    def __init__(self, anchors=()):
        super().__init__()
        self.nl = len(anchors)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

    def forward(self, x):
        self._test_print()
        return x

    def _test_print(self):
        print(self.anchors)


anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
test_Init = test_init(anchors)
input = torch.ones((3, 4))
test_Init(input)