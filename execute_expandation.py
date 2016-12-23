#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:38:57 2016

@author: Yinong

对图像数据进行扩充
"""
from __future__ import print_function

import image_expandation
import os

# 待扩充的数据存储位置
source_dir = '/Users/Yinong/Downloads/data_augmentation/BCtypes_Tongue'

# 扩充后的数据存储的位置
save_dir = '/Users/Yinong/Downloads/data_augmentation/augmentation'

def generate_images(image_list):
    """
    对图像列表中的所有图像进行扩充操作
    """
    for image_path in image_list:
        print('正在处理图像', image_path)
        # 滑动窗口截取操作
        image_expandation.translation(fp=image_path,
                                      save_dir=save_dir,
                                      width=64,
                                      height=64,
                                      stride=4)
        # 水平反转操作
        image_expandation.flip_left_right(fp=image_path,
                                          save_dir=save_dir)
        # 旋转操作
        image_expandation.rotation(fp=image_path,
                                   save_dir=save_dir,
                                   delta_angle=30)
        # 像素浮动操作
        image_expandation.pixel_variation(fp=image_path,
                                          save_dir=save_dir,
                                          bound=5,
                                          positive=True)
        # fancy PCA操作扩充
        image_expandation.fancyPCA(fp=image_path,
                                   save_dir=save_dir,
                                   num=10)
    print('所有扩充操作完成！')

def get_all_image(dir_path):
    """
    获取指定目录下所有的图像数据的路径
    
    Parameters
    ----------
    dir_path: str
        指定需要进行扩充的所有图像存储的目录
    """
    paths = []
    paths.append(dir_path)
    result = []
    while len(paths) > 0:
        temp_dir = paths.pop()
        all_names = os.listdir(temp_dir)
        
        alternatives = []
        for name in all_names:
            if name.startswith('.'):
                continue
            else:
                alternatives.append(os.path.join(temp_dir, name))
        for item in alternatives:
            if os.path.isdir(item):
                paths.append(item)
            if os.path.isfile(item) and (
                item.endswith('.png') or
                item.endswith('.jpg')):
                result.append(item)
    return result

def main():
    image_list = get_all_image(source_dir)
    generate_images(image_list)

if __name__ == '__main__':
    main()
