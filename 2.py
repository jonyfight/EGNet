"""
describe:
@project: 智能广告项目
@author: Jony
@create_time: 2019-07-09 12:21:10
@file: dd.py
"""

import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def load_image(filepath):
    if not os.path.exists(filepath):
        print("File Not Exits")
    im = Image.open(filepath)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label[np.where(label > 0.5)] = 1
    label = label[np.newaxis,...]
    label = label.reshape([366,400])
    plt.imshow(label)
    plt.show()
    # plt.plot(label)
    # plt.draw()
    return label


if __name__=="__main__":
    path = "E:/Smart Image Project/EG-NET/DUTS-TR/DUTS-TR-Mask/ILSVRC2012_test_00000004_edge.png"
    print(load_image(path))
    # plt.plot(path)
    # plt.show()