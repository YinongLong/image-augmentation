#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:36:01 2016

@author: Yinong

图片数据扩展

1. 滑动截取扩展
2. 旋转扩展
3. RGB 浮动扩展
4. fancy PCA

"""
from __future__ import print_function
from __future__ import division

from PIL import Image

import os

import numpy as np

def _construct_path(save_path, name, number, extension):
    """
    生成图片文件存储的具体路径
    """
    number = str(number)
    name = name + '_' + '0' * (4 - len(number)) + number + '.' + extension
    return os.path.join(save_path, name)
    
def _split_image_name(image_file_path):
    """
    从指定的图片位置返回图片的文件名和扩展名
    """
    image_name, extension = os.path.basename(image_file_path).split('.')
    image_name = image_name.split('_')[0]
    return image_name, extension
    
def _get_max_name(dir_path):
    """
    获取生成图片文件夹下最大的文件名
    """
    assert os.path.exists(dir_path), '%s directory is not exist!' % dir_path
    eles = os.listdir(dir_path)
    eles.sort()
    temp_name = eles[-1]
    if not temp_name.startswith('.'):
        num = temp_name.split('.')[0].split('_')[1]
        while num.startswith('0'):
            num = num[1:]
    else:
        num = None
    return num
    
def fancyPCA(fp, num):
    """
    对图像进行fancyPCA进行扩充
    
    fp: str
        图像存储的位置
        
    num: int
        每张图片扩充的个数
    """
    assert os.path.isfile(fp), 'The image file is not exist!'
    
    save_dir, _ = os.path.split(fp)
    
    image = Image.open(fp)
    image_name, extension = _split_image_name(fp)
    
    image_width = image.width
    image_height = image.height
    
    image_data = list(image.getdata())
    image_data = np.array(image_data, dtype=np.float32)[:, :-1]
    # 计算每一维的均值，并减去
    data_mean = np.mean(image_data, axis=0)
    image_data -= data_mean
    # 计算协方差矩阵
    covariance = np.dot(image_data.T, image_data) / (image_data.shape[0]-1)
    assert covariance.shape == (3, 3), 'dimension number is not correct!'
    eigen_val, eigen_vect = np.linalg.eig(covariance)
    
    count = int(_get_max_name(save_dir)) + 1
    
    for _ in range(num):
        alpha = np.random.normal(loc=0.0, scale=0.1, size=(3,))
        weights = alpha * eigen_val
        addition = np.dot(eigen_vect, weights.reshape((-1, 1)))
        
        path = _construct_path(save_dir,
                               image_name,
                               count,
                               extension)
        # 生成扩充的图像
        _support(image,
                 addition[0][0],
                 addition[1][0],
                 addition[2][0],
                 image_width,
                 image_height,
                 path)
        
        count += 1
    image.close()
    print(u'fancy PCA 扩充完成！')
    
def _support(img, r_num, g_num, b_num, width, height, path):
    """
    对图像指定位置的像素进行修改，然后将图片保存到指定路径
    """
    sub_img = img.copy()
    for i in range(height):
        for j in range(width):
            pixel = list(img.getpixel((i, j)))
            pixel[0] += r_num
            pixel[1] += g_num
            pixel[2] += b_num
            pixel[0] = int(pixel[0])
            pixel[1] = int(pixel[1])
            pixel[2] = int(pixel[2])
            sub_img.putpixel((i, j), tuple(pixel))
    sub_img.save(path)
    
def pixel_variation(fp, bound, positive=True):
    """
    对图片的像素进行正向或者负向的浮动
    
    Parameters
    ----------
    fp: str
        图片存储的路径
        
    bound: int
        像素值的浮动范围
        
    positive: bool
        指定像素值浮动的方向，正向或者负向
    """
    assert os.path.isfile(fp), 'The image file is not exist!'
    
    save_dir, _ = os.path.split(fp)
    
    
    image = Image.open(fp)
    
    image_name, extension = _split_image_name(fp)
    
    image_width = image.width
    image_height = image.height
    # 标志正向或者负向
    if positive:
        sign = 1
    else:
        sign = -1
        
    count = int(_get_max_name(save_dir)) + 1
    for i in range(1, bound+1):
        for j in range(1, bound+1):
            for k in range(1, bound+1):
                path = _construct_path(save_dir,
                                   image_name,
                                   count,
                                   extension)
                # 生成图像存储
                _support(image, 
                         sign*i,
                         sign*j,
                         sign*k,
                         image_width,
                         image_height,
                         path)
                count += 1
    image.close()
    print(u'像素浮动扩展结束！')
    
def rotation(fp, delta_angle):
    """
    对指定的图片进行旋转进行扩展
    
    Parameters
    ----------
    fp: str
        指定进行扩展的图片的路径
        
    delta_angle: float
        每次旋转的角度的大小
        
    clockwise: bool, default: False
        指定旋转是按照顺时针或者是逆时针，默认为逆时针
    """
    assert os.path.isfile(fp), 'The image file is not exist!'
    
    save_dir, _ = os.path.split(fp)
    
    count = int(_get_max_name(save_dir)) + 1
    
    image = Image.open(fp)
    
    image_name, extension = _split_image_name(fp)
    
    step = 1
    angle = step * delta_angle
    
    while angle <= 360:
        sub_image = image.rotate(angle)
        sub_image.save(_construct_path(save_dir,
                                       image_name,
                                       count,
                                       extension))
        count += 1
        step += 1
        angle = step * delta_angle
    image.close()
    print(u'图片的旋转扩展结束！')
    
def flip_left_right(fp):
    """
    对指定的图片文件进行水平的对折
    
    Parameters
    ----------
    fp: str
        指定进行扩展的图片的路径
    """
    assert os.path.isfile(fp), 'The image file is not exist!'
    
    save_dir, _ = os.path.split(fp)
    
    count = int(_get_max_name(save_dir)) + 1
    
    image = Image.open(fp)
    new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    image_name, extension = _split_image_name(fp)
    new_image.save(_construct_path(save_dir,
                                   image_name,
                                   count,
                                   extension))
    image.close()
    print(u'水平对折扩展结束！')

def translation(fp, width, height, stride=1):
    """
    对图片进行滑动窗口平移扩展
    
    Parameters
    ----------
    fp: str
        指定图片存储的路径
        
    width: int
        滑动窗口的宽度，必须小于等于图片本身的宽度
        
    height: int
        滑动窗口的高度，必须小于等于图片本身的高度
        
    stride: int
        滑动窗口移动的步幅大小
    """
    assert os.path.isfile(fp), 'The image file is not exist!'
    
    save_dir, _ = os.path.split(fp)
    
    image = Image.open(fp)
    
    image_width = image.width
    image_height = image.height
            
    image_name, extension = _split_image_name(fp)
    
    upper = 0; lower = height
    
    count = int(_get_max_name(save_dir)) + 1
    
    while lower <= image_height:
        left = 0; right = width
        while right <= image_width:
            box = (left, upper, right, lower)
            
            sub_image = image.crop(box)
            
            save_path = _construct_path(save_dir,
                                        image_name,
                                        count,
                                        extension)
            
            sub_image.save(save_path)
            
            count += 1
            left += stride
            right += stride
        upper += stride; lower += stride
    print(u'滑动窗平移扩展结束！')