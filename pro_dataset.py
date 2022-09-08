import os
import torch
import PIL
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import cv2
import json


class whu_dataset(Dataset):
    def __init__(self):
        super(whu_dataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


dataset_root_dir = '/home/xcy/dataset/chusai_release'
train_root_dir = os.path.join(dataset_root_dir, 'train')
json_name = 'instances_train.json'
json_path = os.path.join(train_root_dir, json_name)
imgs_path = os.path.join(train_root_dir, 'images')
whu_dataset = COCO(json_path)
imgs_id = whu_dataset.getImgIds()
img_id = imgs_id[10]
img_info = whu_dataset.loadImgs([img_id])
