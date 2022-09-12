import torch.nn as nn


class test_init(nn.Module):

    def __init__(self, anchors=()):
        self.nl = len(anchors)
        self._test_print()

    def _test_print(self):
        print(self.anchors)


test_Init = test_init([116, 90, 156, 198, 373, 326])