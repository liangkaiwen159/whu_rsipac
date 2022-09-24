import os
import PIL
import torch
from PIL import Image, ImageDraw
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import glob
# 1: uncross 2: cross
dataset_root_dir = '/mnt/users/datasets/chusai_crop'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class Whu_dataset(Dataset):

    def __init__(self, dataset_root_dir, cache=True):
        super(Whu_dataset, self).__init__()
        self.dataset_root_dir = dataset_root_dir
        self.train_root_dir = os.path.join(dataset_root_dir, 'train')
        self.imgs_path = os.path.join(self.train_root_dir, 'images')
        self.mask_imgs_path = os.path.join(self.train_root_dir, 'masks')
        self.lables_path = os.path.join(self.train_root_dir, 'lables')
        self.imgs_list = []
        self.mask_imgs_list = []
        self.lables_list = []
        self.out_lables = []
        self.lables = []
        self.cache = cache
        for dirpath, dirnames, filenames in os.walk(self.imgs_path):
            for filename in filenames:
                self.imgs_list.append(filename)
                self.mask_imgs_list.append(filename.replace('jpg', 'png'))
                if os.path.exists(os.path.join(self.lables_path, filename.replace('jpg', 'txt'))):
                    self.lables_list.append(os.path.join(self.lables_path, filename.replace('jpg', 'txt')))
                else:
                    self.lables_list.append(None)
        for i in range(len(self.lables_list)):
            lable_txt = self.lables_list[i]
            if lable_txt:
                f = open(lable_txt, 'r')
                lables = f.readlines()
                for lable in lables:
                    _lable = []
                    lable = lable.split('\n')[0]
                    it_cls, x1, y1, x2, y2 = [int(x) for x in lable.split(' ')]
                    center_x = (x1 + x2) / 640
                    center_y = (y1 + y2) / 640
                    w = (x2 - x1) / 640
                    h = (y2 - y1) / 640
                    self.lables.append([it_cls, center_x, center_y, w, h])
                    _lable.append([it_cls, center_x, center_y, w, h])
                self.out_lables.append(_lable)
            else:
                self.out_lables.append(None)
        self.out_lables = np.array(self.out_lables, dtype=object)
        n = len(self.imgs_list)
        if self.cache:
            gb = 0
            self.img_hw0, self.img_hw = [None] * n, [None] * n
        # print(type(self.out_lables))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        to_tensor = torchvision.transforms.ToTensor()
        img = Image.open(os.path.join(self.imgs_path, self.imgs_list[index]))
        img = to_tensor(img)
        mask_img = Image.open(os.path.join(self.mask_imgs_path, self.mask_imgs_list[index]))
        mask_img = to_tensor(mask_img)

        return img, mask_img, self.out_lables[index]

    @staticmethod
    def col_fun(batch):
        img = []
        mask_img = []
        targets = []
        for sample in batch:
            img.append(sample[0])
            mask_img.append(sample[1])
            targets.append(0 if sample[2] is None else sample[2])
        return torch.stack(img, 0), torch.stack(mask_img, 0), targets


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        # img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img0

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap


class Whu_sub_dataset(Whu_dataset):

    def __init__(self, dataset_root_dir, indexs):
        super().__init__(dataset_root_dir)
        _imgs = []
        _masks = []
        _lables = []
        for index in indexs:
            _imgs.append(self.imgs_list[index])
            _masks.append(self.mask_imgs_list[index])
            _lables.append(self.out_lables[index])
        self.imgs_list = _imgs
        self.mask_imgs_list = _masks
        self.out_lables = _lables

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        to_tensor = torchvision.transforms.ToTensor()
        img = Image.open(os.path.join(self.imgs_path, self.imgs_list[index]))
        img = to_tensor(img)
        mask_img = Image.open(os.path.join(self.mask_imgs_path, self.mask_imgs_list[index]))
        mask_img = to_tensor(mask_img)

        return img, mask_img, self.out_lables[index]


def creat_val_loader(dataset_root_path, batch_size=1, num_workers=1):
    np.random.seed(0)
    whu_dataset = Whu_dataset(dataset_root_path, cache=True)
    train_dataset_size = int(len(whu_dataset) * 0.9)
    train_dataset_indexs = np.random.choice(np.arange(len(whu_dataset)), train_dataset_size, replace=False)
    val_dataset_indexs = np.setdiff1d(np.arange(len(whu_dataset)), train_dataset_indexs)
    val_dataset = Whu_sub_dataset(dataset_root_path, val_dataset_indexs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return val_loader


if __name__ == '__main__':
    np.random.seed(0)
    my_dataset = Whu_dataset(dataset_root_dir)
    my_dataset_loader = DataLoader(my_dataset, 1, True, collate_fn=Whu_dataset.col_fun)
    total_img = len(my_dataset)
    train_dataset_size = int(total_img * 0.9)
    val_dataset_size = total_img - train_dataset_size
    # _train_dataset, _val_dataset = random_split(my_dataset, [train_dataset_size, val_dataset_size],
    #                                             generator=torch.Generator().manual_seed(0))
    train_dataset_indexs = np.random.choice(np.arange(total_img), train_dataset_size, replace=False)
    val_dataset_indexs = np.setdiff1d(np.arange(total_img), train_dataset_indexs)
    train_dataset = Whu_sub_dataset(dataset_root_dir, train_dataset_indexs)
    val_dataset = Whu_sub_dataset(dataset_root_dir, val_dataset_indexs)
    train_data_loader = DataLoader(train_dataset, 10, collate_fn=Whu_dataset.col_fun)
    # for i in range(len(train_dataset)):
    #     if train_dataset[i][2]:
    #         print(train_dataset[i][2])
    #         break
    for imgs, masks, lables in train_data_loader:
        print(lables)
        break