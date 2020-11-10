# author: 
# contact: onlydgi@foxmail.com
# datetime:2020/7/15 上午1:47
# software: PyCharm
"""
文件说明：
"""
import math

import cv2
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
# ia.seed(3)

import matplotlib.pyplot as plt


def resize_pos(x1, y1, src_size, tar_size):
    """
    :param x1:
    :param y1:
    :param src_size: width * hight
    :param tar_size: width * hight
    :return:
    """
    w1 = src_size[1]
    h1 = src_size[0]
    w2 = tar_size[0]
    h2 = tar_size[1]
    y1 = np.array(y1).astype(np.float32)
    x1 = np.array(x1).astype(np.float32)
    h1 = np.array(h1).astype(np.float32)
    w1 = np.array(w1).astype(np.float32)
    h2 = np.array(h2).astype(np.float32)
    w2 = np.array(w2).astype(np.float32)
    # print("y1:", y1,"h1", h1,"h2", h2)
    # print("x1", x1,"w1" ,w1,"w2", w2)
    y2 = np.int(np.round(np.float64(y1 / h1 * h2)))
    x2 = np.int(np.round(np.float64(x1 / w1 * w2)))
    # print(x2, y2)
    return x2, y2


def makeHeatmap_1(img,vert_posi, disc_posi):
    """

    :param img:
    :param vert_posi:
    :param disc_posi:
    :return: resize the img and make the heatmap using imgaug.resize() and aug_det.augment_keypoints.
    """
    keyposi = vert_posi + disc_posi
    kpsoi_ia = ia.KeypointsOnImage.from_xy_array(keyposi, shape=(img.shape[0], img.shape[1], 1))

    aug = iaa.Sequential([iaa.Resize(size={'height': 256, 'width': 256})])
    aug_det = aug.to_deterministic()

    img_data = img[:, :, np.newaxis]
    image_aug = aug_det.augment_image(img_data)
    kpsoi_aug = aug_det.augment_keypoints(kpsoi_ia)
    image_aug_2D = np.squeeze(image_aug, 2)

    distance_maps = kpsoi_aug.to_distance_maps()
    height, width = kpsoi_aug.shape[0:2]
    max_distance = np.linalg.norm(np.float32([0, 0]) - np.float32([height, width]))
    distance_maps_normalized = distance_maps / max_distance
    distance_maps_normalized = 1.0 - distance_maps_normalized
    keyposi_list = kpsoi_aug.to_xy_array()
    return image_aug_2D, keyposi_list, distance_maps_normalized


def makeHeatmap(img, origi_keyposi):
    # keyposi = vert_posi + disc_posi
    # point posi is sorted from up to down
    # keyposi = vert_posi
    # pred 10 point
    new_keyposi = []
    heatmap = []
    for point in origi_keyposi:
        heatmap_img_one = makeGaussianMap(img, point, sigma=5, half_sz=20)
        resize_heatmap_img_one = cv2.resize(heatmap_img_one, (256, 256))
        resize_heatmap_img_one = np.array(resize_heatmap_img_one)
        heatmap.append(resize_heatmap_img_one)
        new_keyposi.append(np.array(resize_pos(point[0], point[1], img.shape, (256, 256))))
    img = cv2.resize(img, (256, 256))
    heatmap = np.array(heatmap)
    new_keyposi = np.array(new_keyposi)
    return img, new_keyposi, heatmap


def makeGaussianMap(img, center, sigma=20, half_sz=50):

    """
    Parameters
    -heatmap: 热图（heatmap）
    - plane_idx：关键点列表中第几个关键点（决定了在热图中通道）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
    # oneHeatmap = np.zeros(img.shape)
    center_x, center_y = center  # mou发
    height, width = img.shape

    th = 4.6052
    delta = np.sqrt(th * 2)

    x0 = int(max(0, center_x - half_sz + 0.5))
    y0 = int(max(0, center_y - half_sz + 0.5))

    x1 = int(min(width - 1, center_x + half_sz + 0.5))
    y1 = int(min(height - 1, center_y + half_sz + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    # arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    heatmap = np.zeros(img.shape)
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    heatmap[y0:y1 + 1, x0:x1 + 1] = arr_exp
    #  show time
    # plt.imshow(heatmap)
    # plt.show()
    # show_img = img*heatmap
    # plt.imshow(show_img)
    # plt.show()
    return heatmap

# heatmap=makeGaussianMap(img,(135,170),10,30)




if __name__ == '__main__':

    label = data[1] + data[3]
    kpsoi_new = ia.KeypointsOnImage.from_xy_array(label, shape=(346, 384, 1))
    aug = iaa.Sequential([iaa.Scale(size={'height': 256, 'width': 256})])

    #  (640, 640) (512, 512)  (480, 480)

    aug_det = aug.to_deterministic()
    newdata = data[0][:, :, np.newaxis]
    image_aug = aug_det.augment_image(newdata)
    kpsoi_aug = aug_det.augment_keypoints(kpsoi_new)

    ia.imshow(kpsoi_aug.draw_on_image(np.squeeze(image_aug), size=3, color=255))

    distance_maps = kpsoi_aug.to_distance_maps()
    print("Image shape:", distance_maps.shape)
    print("Distance maps shape:", distance_maps.shape)
    print("Distance maps dtype:", distance_maps.dtype)
    print("Distance maps min:", distance_maps.min(), "max:", distance_maps.max())


    height, width = kpsoi_aug.shape[0:2]
    max_distance = np.linalg.norm(np.float32([0, 0]) - np.float32([height, width]))
    distance_maps_normalized = distance_maps / max_distance
    print("min:", distance_maps.min(), "max:", distance_maps_normalized.max())

    # ————————————————
    # 版权声明：本文为CSDN博主「XerCis」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/lly1122334/article/details/88874502
