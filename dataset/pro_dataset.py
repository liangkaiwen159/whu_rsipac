import os
import PIL
import torch
from PIL import Image, ImageDraw
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
# 1: uncross 2: cross
dataset_root_dir = 'E:\梁凯文\datasets\chusai_crop'


class whu_dataset(Dataset):
    def __init__(self, dataset_root_dir):
        super(whu_dataset, self).__init__()
        self.dataset_root_dir = dataset_root_dir
        self.train_root_dir = os.path.join(dataset_root_dir, 'train')
        self.imgs_path = os.path.join(self.train_root_dir, 'images')
        self.mask_imgs_path = os.path.join(self.train_root_dir, 'masks')
        self.lables_path = os.path.join(self.train_root_dir, 'lables')
        self.imgs_list = []
        self.mask_imgs_list = []
        self.lables_list = []
        self.out_lables = []
        for dirpath, dirnames, filenames in os.walk(self.imgs_path):
            for filename in filenames:
                self.imgs_list.append(filename)
                self.mask_imgs_list.append(filename.replace('jpg', 'png'))
                if os.path.exists(os.path.join(self.lables_path, filename.replace('jpg', 'txt'))):
                    self.lables_list.append(
                        os.path.join(self.lables_path, filename.replace('jpg', 'txt')))
                else:
                    self.lables_list.append(None)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        to_tensor = torchvision.transforms.ToTensor()
        img = Image.open(os.path.join(self.imgs_path, self.imgs_list[index]))
        img = to_tensor(img)
        mask_img = Image.open(os.path.join(
            self.mask_imgs_path, self.mask_imgs_list[index]))
        mask_img = to_tensor(mask_img)
        lable_txt = self.lables_list[index]
        self.out_lables = []
        if lable_txt:
            f = open(lable_txt, 'r')
            lables = f.readlines()
            for lable in lables:
                lable = lable.split('\n')[0]
                self.out_lables.append([int(x) for x in lable.split(' ')])
        return img, mask_img, self.out_lables

    @staticmethod
    def col_fun(batch):
        img = []
        mask_img = []
        targets = []
        for sample in batch:
            img.append(sample[0])
            mask_img.append(sample[1])
            targets.append(torch.FloatTensor(sample[2]))
        return torch.stack(img, 0), torch.stack(mask_img, 0), targets


my_dataset = whu_dataset(dataset_root_dir)
my_dataset_loader = DataLoader(
    my_dataset, 2, True, collate_fn=whu_dataset.col_fun)
