#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:36:01 2016

@author: Yinong

图片数据扩展

1. 滑动截取扩展
2. 旋转扩展
3. RGB 浮动扩展

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
    
def fancyPCA(fp, save_dir, num):
    """
    对图像进行fancyPCA进行扩充
    
    fp: str
        图像存储的位置
        
    save_dir: str
        扩充的图像保存的根目录
        
    num: int
        每张图片扩充的个数
    """
    assert os.path.isfile(fp), '图像不存在！'
    
    operation_dir = 'fancyPCA'
    save_dir = os.path.join(save_dir, operation_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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
    assert covariance.shape == (3, 3), '计算出现问题！'
    eigen_val, eigen_vect = np.linalg.eig(covariance)
    for i in range(num):
        alpha = np.random.normal(loc=0.0, scale=0.1, size=(3,))
        weights = alpha * eigen_val
        addition = np.dot(eigen_vect, weights.reshape((-1, 1)))
        
        path = _construct_path(save_dir,
                               image_name,
                               i+1,
                               extension)
        # 生成扩充的图像
        _support(image,
                 addition[0][0],
                 addition[1][0],
                 addition[2][0],
                 image_width,
                 image_height,
                 path)
    image.close()
    print('fancy PCA 扩充完成！')
    
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
    
def pixel_variation(fp, save_dir, bound, positive=True):
    """
    对图片的像素进行正向或者负向的浮动
    
    Parameters
    ----------
    fp: str
        图片存储的路径
        
    save_dir: str
        扩展图片存储的根目录
        
    bound: int
        像素值的浮动范围
        
    positive: bool
        指定像素值浮动的方向，正向或者负向
    """
    assert os.path.isfile(fp), '图片文件不存在！'
    
    operation_dir = 'pixel_variation'
    save_dir = os.path.join(save_dir, operation_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = Image.open(fp)
    
    image_name, extension = _split_image_name(fp)
    
    image_width = image.width
    image_height = image.height
    # 标志正向或者负向
    if positive:
        sign = 1
    else:
        sign = -1
        
    image_count = 1
    for i in range(1, bound+1):
        for j in range(1, bound+1):
            for k in range(1, bound+1):
                path = _construct_path(save_dir,
                                   image_name,
                                   image_count,
                                   extension)
                # 生成图像存储
                _support(image, 
                         sign*i,
                         sign*j,
                         sign*k,
                         image_width,
                         image_height,
                         path)
                image_count += 1
    image.close()
    print('像素浮动扩展结束！')
    
def rotation(fp, save_dir, delta_angle):
    """
    对指定的图片进行旋转进行扩展
    
    Parameters
    ----------
    fp: str
        指定进行扩展的图片的路径
        
    save_dir: str
        保存生成图片的根目录
        
    delta_angle: float
        每次旋转的角度的大小
        
    clockwise: bool, default: False
        指定旋转是按照顺时针或者是逆时针，默认为逆时针
    """
    assert os.path.isfile(fp), '图片文件不存在！'
    
    operation_dir = 'rotation'
    save_dir = os.path.join(save_dir, operation_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = Image.open(fp)
    
    image_name, extension = _split_image_name(fp)
    count = 1
    angle = count * delta_angle
    while angle <= 360:
        sub_image = image.rotate(angle)
        sub_image.save(_construct_path(save_dir,
                                       image_name,
                                       count,
                                       extension))
        count += 1
        angle = count * delta_angle
    image.close()
    print('图片的旋转扩展结束！')
    
def flip_left_right(fp, save_dir):
    """
    对指定的图片文件进行水平的对折
    
    Parameters
    ----------
    fp: str
        指定进行扩展的图片的路径
        
    save_dir: str
        生成图片的存储目录
    """
    assert os.path.isfile(fp), '图片文件不存在！'
    
    operation_dir = 'flip_left_right'
    save_dir = os.path.join(save_dir, operation_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = Image.open(fp)
    new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    image_name, extension = _split_image_name(fp)
    new_image.save(_construct_path(save_dir,
                                   image_name,
                                   1,
                                   extension))
    image.close()
    print('水平对折扩展结束！')

def translation(fp, save_dir, width, height, stride=1):
    """
    对图片进行滑动窗口平移扩展
    
    Parameters
    ----------
    fp: str
        指定图片存储的路径
        
    save_dir: str
        所有改种扩充的图片存储的根目录
        
    width: int
        滑动窗口的宽度，必须小于等于图片本身的宽度
        
    height: int
        滑动窗口的高度，必须小于等于图片本身的高度
        
    stride: int
        滑动窗口移动的步幅大小
    """
    assert os.path.isfile(fp), '图片文件不存在！'
    
    operation_dir = 'translation'
    save_dir = os.path.join(save_dir, operation_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = Image.open(fp)
    image_width = image.width
    image_height = image.height
    assert (image_width >= width and 
            image_height >= height and
            width > 0 and
            height > 0), '设置的参数不合适！'
    
    image_name, extension = _split_image_name(fp)
    
    upper = 0; lower = height
    count = 1
    while lower <= image_height + 1:
        left = 0; right = width
        while right <= image_width + 1:
            box_bound = (left, upper, right, lower)
            sub_image = image.crop(box_bound)
            
            sub_image.save(_construct_path(save_dir,
                                          image_name,
                                          count,
                                          extension))
            count += 1
            left += stride
            right += stride
        upper += stride; lower += stride
    image.close()
    print('滑动窗平移扩展结束！')