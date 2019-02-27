# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:32:10 2018

@author: h439
"""

import os
import random
import shutil

if __name__ == '__main__':
    random.seed(2018)
    
    train_file = './datasets/train.txt'
    train_dir = './datasets/train/'
    val_dir = './datasets/valid/'
    
    #获取训练文件
    train_class2Image = {}
    with open(train_file,'r') as file:
        contents = file.readlines()
        for content in contents:
            content = content.replace('\n','')
            content = content.split(' ')
            file, classes = content[0], content[1]
            
#            Python 字典 keys() 方法以列表返回一个字典所有的键。
            if classes not in train_class2Image.keys():
                train_class2Image[classes] = []
            train_class2Image[classes].append(file)
            
        for index, key in enumerate(train_class2Image.keys()):
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
#同时列出数据和数据下标，一般用在 for 循环当中
            print('%d/%d, dir:%s' % (index, len(train_class2Image.keys()),key), end = '\r')
            
            #创建新文件夹，key为1到100
            dirPath = os.path.join(train_dir, key)
            valPath = os.path.join(val_dir, key)
            
            #训练集
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            #训练集
            if not os.path.exists(valPath):
                os.makedirs(valPath)
            #移动文件
            for train_image in train_class2Image[key]:
                origin_path = os.path.join(train_dir, train_image)
                if random.random() > 0:
                    dest_path = os.path.join(dirPath, train_image)
                else:
                    dest_path = os.path.join(valPath, train_image)
                shutil.move(origin_path, dest_path)
