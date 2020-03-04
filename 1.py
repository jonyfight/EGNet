"""
describe:
@project: 智能广告项目
@author: Jony
@create_time: 2019-07-09 12:21:10
@file: dd.py
"""

import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import torch


class ImageDataTrain(data.Dataset):
    def __init__(self):
        self.sal_source = "E:/Smart Image Project/EG-NET/DUTS-TR/train_pair_edge.lst"
        self.sal_root = "E:/Smart Image Project/EG-NET/DUTS-TR"

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
            # sal_list = str(sal_list)
        self.sal_num = len(self.sal_list)


    def __getitem__(self, item):

        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_label = load_sal_label(os.path.join(self.sal_root,self.sal_list[item % self.sal_num].split()[1]))
        sal_edge = load_edge_label(os.path.join(self.sal_root,self.sal_list[item % self.sal_num].split()[2]))
        sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)
        sample = {'sal_image':sal_image, 'sal_label':sal_label, 'sal_edge':sal_edge}
        # sal_num = len(sal_list)
        # return type(sal_list)
        return sample


    def __len__(self):
        return self.sal_num


def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0,1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
        edge = edge[:, :, ::-1].copy()
        return img,label,edge


def cv_random_crop_flip(img, label, resize_size, crop_size, random_flip=True):
    def get_params(img_size,output_size):
        h,w = img_size
        th , tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0,h-th)
        j = random.randint(0,w-tw)
        return i,j,th,tw
    if random_flip:
        flip_flag = random.randint(0,1)
    img = img.transpose((1,2,0))
    label = label[0,:,:]
    img = cv2.resize(img, (resize_size[1], resize_size[0]),interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    i,j,h,w = get_params(resize_size,crop_size)
    img = img[i:i+h, j:j+w, :].transpose((2,0,1))
    label = label[i:i+h, j:j+w][np.newaxis,...]
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img,label


def load_image(pah):
    if not os.path.exists(pah):
        print("File Not Exists")
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699,116.66877,122.67892))
    in_ = in_.transpose((2,0,1))
    return in_


def load_edge_label(pah):
    if not os.path.exists(pah):
        print("File Not Exists")
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label


def load_sal_label(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


if __name__=="__main__":
    i = ImageDataTrain()
    print(i.sal_source)
# if __name__=="__main__":
#     sal_root = "E:/Smart Image Project/EG-NET/DUTS-TR"
#     path = "E:/Smart Image Project/EG-NET/DUTS-TR/train_pair_edge.lst"
#     print(ImageDataTrain(path))
    # print(cv_random_flip(ImageDataTrain(path)))