from lib2to3.pytree import convert
import os
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

dataset_root_dir = '/home/xcy/dataset/chusai_release'
dataset_root_dir = dataset_root_dir
train_root_dir = os.path.join(dataset_root_dir, 'train')
json_path = os.path.join(train_root_dir, 'instances_train.json')
imgs_path = os.path.join(train_root_dir, 'images')
whu_dataset = COCO(json_path)
whu_dataset_classes = dict([(v['id'], v['name']) for k, v in whu_dataset.cats.items()])
imgs_id = whu_dataset.getImgIds()
crop_dataset_root_dir = '/home/xcy/dataset/chusai_crop'
crop_train_dir = os.path.join(crop_dataset_root_dir, 'train')
crop_imgs_dir = os.path.join(crop_train_dir, 'images')
crop_masks_dir = os.path.join(crop_train_dir, 'masks')
crop_lables_dir = os.path.join(crop_train_dir, 'lables')
center_points_x = []
center_points_y = []
widths = []
heights = []
imgs_list = []
mask_imgs_list = []


def crop_img(img_path, targets, size=640, gap=160):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    number_w = (img_w - gap) // (size - gap)
    number_h = (img_h - gap) // (size - gap)
    x_points = np.arange(number_w) * (size - gap)
    y_points = np.arange(number_h) * (size - gap)
    # if img_w - x_points[-1] > size:
    x_points = np.append(x_points, img_w - size)
    y_points = np.append(y_points, img_h - size)
    x1_points = (x_points + size)
    y1_points = (y_points + size)
    X_sequence, Y_sequence = np.meshgrid(x_points, y_points)
    X1_sequence, Y1_sequence = np.meshgrid(x1_points, y1_points)
    X_sequence = X_sequence.reshape(number_h + 1, number_w + 1, 1)  # 裁剪区域左上角点X坐标
    Y_sequence = Y_sequence.reshape(number_h + 1, number_w + 1, 1)  # 裁剪区域左上角点Y坐标
    X1_sequence = X1_sequence.reshape(number_h + 1, number_w + 1, 1)  # 裁剪区域右下角点X坐标
    Y1_sequence = Y1_sequence.reshape(number_h + 1, number_w + 1, 1)  # 裁剪区域右下角点X坐标
    XY_sequence = np.concatenate((X_sequence, Y_sequence, X1_sequence, Y1_sequence),
                                 -1)  # 矩阵形式的每个左上角顶点对应的左上角点和右下角点坐标 (M, N, 4)
    segmentation_boxes = XY_sequence.reshape(-1, 4)
    index_x_list = []
    index_y_list = []
    convert_targets_xy = []
    convert_targets_cls = []
    watch_img = np.zeros((img_h + 50, img_w + 50, 3), np.uint8)
    watch_img.fill(255)
    # 遍历标签寻找对应的裁剪框，并限位
    for target in targets:
        x, y, w, h = target['bbox']
        cls = target['category_id']
        center_x = x + int(w / 2)
        center_y = y + int(h / 2)
        x1 = x + w
        y1 = y + h
        cv2.rectangle(watch_img, (x, y), (x1, y1), (255, 0, 0), 5)  # 绘制真实标注
        index_x = center_x // (size - gap) if center_x // (size - gap) <= number_w else number_w
        index_y = center_y // (size - gap) if center_y // (size - gap) <= number_h else number_h
        convert_targets_cls.append(cls)
        convert_targets_xy.append([
            max(x, XY_sequence[index_y][index_x][0]),
            max(y, XY_sequence[index_y][index_x][1]),
            min(x1, XY_sequence[index_y][index_x][2]),
            min(y1, XY_sequence[index_y][index_x][3])
        ])
        index_x_list.append(index_x)
        index_y_list.append(index_y)
    index_x_list = np.array(index_x_list).reshape(len(targets), 1)
    index_y_list = np.array(index_y_list).reshape(len(targets), 1)
    index_xy_list = np.concatenate((index_x_list, index_y_list), axis=-1)
    # 遍历裁切，限位
    for i in range(number_w + 1):
        for j in range(number_h + 1):
            cv2.circle(watch_img, XY_sequence[j][i][:2], 10, (255, 0, 0), 20)
            cv2.rectangle(watch_img, XY_sequence[j][i][:2], XY_sequence[j][i][2:], (0, 255, 0), 5)
            write_img = img[XY_sequence[j][i][1]:XY_sequence[j][i][3], XY_sequence[j][i][0]:XY_sequence[j][i][2], :]
            write_path = Path(
                (img_path.replace('release', 'crop').split('.')[0] + '_' + str(j) + '_' + str(i) + '.jpg').replace(
                    '\\', "/"))
            write_mask_img = cv2.imdecode(
                np.fromfile(img_path.replace('images', 'masks').replace('tif', 'png'), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)[XY_sequence[j][i][1]:XY_sequence[j][i][3],
                                      XY_sequence[j][i][0]:XY_sequence[j][i][2]]
            write_mask_path = Path(str(write_path).replace('images', 'masks').replace('jpg', 'png'))
            cv2.imencode('.jpg', write_img)[1].tofile(write_path)
            cv2.imencode('.png', write_mask_img)[1].tofile(write_mask_path)

    for (i, j) in index_xy_list:
        cv2.rectangle(watch_img, XY_sequence[j][i][:2], XY_sequence[j][i][2:], (0, 0, 255), 20)

    for count, xy in enumerate(convert_targets_xy):
        cv2.rectangle(watch_img, xy[:2], xy[2:], (0, 0, 0), 20)
        i, j = index_xy_list[count]
        txt_path = (img_path.replace('release', 'crop').split('.')[0] + '_' + str(j) + '_' + str(i) + '.jpg').replace(
            '\\', "/").replace('jpg', 'txt').replace('images', 'lables')
        cls = convert_targets_cls[count]
        x = xy[0] - XY_sequence[j][i][0]
        y = xy[1] - XY_sequence[j][i][1]
        x1 = xy[2] - XY_sequence[j][i][0]
        y1 = xy[3] - XY_sequence[j][i][1]
        # 将相对标签写入txt文件, 格式类别，左上角点，右下角点
        lables_txt = open(txt_path, 'a+')
        lables_txt.writelines([str(cls), ' ', str(x), ' ', str(y), ' ', str(x1), ' ', str(y1), '\n'])
        lables_txt.close()
    # cv2.namedWindow('result', 0)
    # # cv2.resizeWindow('result', 1440, 900)
    # cv2.imshow('result', watch_img)
    # cv2.waitKey(0)


for img_id in tqdm(imgs_id):
    # 获取id对应图像所有的annotations idx信息
    ann_ids = whu_dataset.getAnnIds(img_id)

    # 根据annotations idx信息获取标注信息
    targets = whu_dataset.loadAnns(ann_ids)
    # 获取图像名称
    img_name = whu_dataset.loadImgs(img_id)[0]['file_name']
    # print(img_name)
    img_path = os.path.join(imgs_path, img_name)
    mask_img_path = os.path.join(train_root_dir, 'masks', img_name.replace('tif', 'png'))
    imgs_list.append(img_path)
    mask_imgs_list.append(mask_img_path)

    # # 查看图像及标注
    # img = Image.open(os.path.join(train_root_dir, 'images',
    #                               img_name)).convert('RGB')
    # draw = ImageDraw.Draw(img)
    # for target in targets:
    #     x, y, w, h = target['bbox']
    #     x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
    #     draw.rectangle((x1, y1, x2, y2), outline='red')
    #     draw.text((x1, y1), whu_dataset_classes[target['category_id']])
    # plt.imshow(img)
    # plt.show()

    # 查看bbox的宽高和中心点分布
    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1 = int(x + w / 2), int(y + h / 2)
        center_points_x.append(x1)
        center_points_y.append(y1)
        widths.append(w)
        heights.append(h)
    crop_img(img_path, targets)
# 图看分布
# plt.scatter(center_points_x, center_points_y, s=1)
# plt.show()
# plt.scatter(widths, heights, s=1)
# plt.show()
