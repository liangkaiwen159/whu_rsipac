import os
import torch
import PIL
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import cv2


class whu_dataset(Dataset):
    def __init__(self):
        super(whu_dataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


dataset_root_dir = 'E:\梁凯文\datasets\chusai_release'
train_root_dir = os.path.join(dataset_root_dir, 'train')
json_name = 'instances_train.json'
with open(os.path.join(train_root_dir, json_name), 'r') as f:
    lines = f.readlines()
f.close()
print(len(lines))
