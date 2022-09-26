import cv2
import numpy as np
from pathlib import Path
from PIL.Image import open


def crop_img(img, size=640, gap=0):  #img rgb
    img_h, img_w, _ = img.shape
    number_w = (img_w - gap) // (size - gap)
    number_h = (img_h - gap) // (size - gap)
    x_points = np.arange(number_w) * (size - gap)
    y_points = np.arange(number_h) * (size - gap)
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
    watch_img = np.zeros((img_h + 50, img_w + 50, 3), np.uint8)
    watch_img.fill(255)
    croped_imglist = {}
    # 遍历标签寻找对应的裁剪框，并限位
    # 遍历裁切，限位
    for i in range(number_w + 1):
        for j in range(number_h + 1):
            #     cv2.circle(watch_img, XY_sequence[j][i][:2], 10, (255, 0, 0), 20)
            #     cv2.rectangle(watch_img, XY_sequence[j][i][:2], XY_sequence[j][i][2:], (0, 255, 0), 5)
            write_img = img[XY_sequence[j][i][1]:XY_sequence[j][i][3], XY_sequence[j][i][0]:XY_sequence[j][i][2], :]
            write_img = write_img.transpose((2, 0, 1))
            img_index = '%d_%d' % (j, i)
            croped_imglist[img_index] = write_img
    return croped_imglist, XY_sequence


def cat_img(ori_shape, img_list, XY_sequence, size=640, gap=0, mask=False):
    len_w = XY_sequence.shape[1]
    len_h = XY_sequence.shape[0]
    # 遍历标签寻找对应的裁剪框，并限位
    # 遍历裁切，限位
    img = np.zeros((ori_shape[1], ori_shape[2], ori_shape[0] if not mask else 1))
    for i in range(len_w):
        for j in range(len_h):
            #     cv2.circle(watch_img, XY_sequence[j][i][:2], 10, (255, 0, 0), 20)
            #     cv2.rectangle(watch_img, XY_sequence[j][i][:2], XY_sequence[j][i][2:], (0, 255, 0), 5)
            img[XY_sequence[j][i][1]:XY_sequence[j][i][3],
                XY_sequence[j][i][0]:XY_sequence[j][i][2], :] = img_list['%d_%d' % (j, i)].transpose(
                    (1, 2, 0))[..., ::-1]
    return img


if __name__ == "__main__":
    test_img_path = '/home/xcy/dataset/chusai_crop/test/images/400.tif'
    img = cv2.imread(test_img_path)
    print(img.shape)
    # cv2.imwrite('before.jpg', img)
    crop_img_list, XY_sequence = crop_img(img)
    img = cat_img(crop_img_list, XY_sequence)
    # cv2.imwrite('after.jpg', img)
    print(img.shape)
