# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:51:26 2018

@author: h439
"""

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3_matt import InceptionV3, preprocess_input
#from keras.applications.vgg16 import VGG16, preprocess_input preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2 ,preprocess_input
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.xception import Xception
#from keras.applications.densenet import DenseNet169,preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad, SGD
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

#数据准备
IM_WIDTH, IM_HEIGHT = 299,299 #InceptionV3指定的图片尺寸
FC_SIZE = 1024

train_dir = './datasets/train'
val_dir = './datasets/valid'
nb_classes = 100
nb_epoch = 99
batch_size = 8

def get_nb_files(directory):#得到文件的数量，个数
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

nb_train_samples = get_nb_files(train_dir) #训练集样本个数
nb_class = len(glob.glob(train_dir + "/*")) #分类数
ub_val_samples = get_nb_files(val_dir) #验证集样本个数
nb_epoch = int(nb_epoch) 
batch_size = int(batch_size)

#图片生成器
train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,#对应模型的图片预处理
        rotation_range = 30,
        width_shift_range = 0.1,
        height_shift_range = 0.2,
        shear_range = 0.1,
        zoom_range = 0.3,
#        horizontal_flip = True,#裁剪
        )

val_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        )

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (IM_WIDTH, IM_HEIGHT),
        batch_size = batch_size,
        class_mode = 'categorical',#多分类，二分类为别的
        )
#print(train_generator.nb_class)

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size = (IM_WIDTH, IM_HEIGHT),
        batch_size = batch_size,
        class_mode = 'categorical',#多分类，二分类为别的
        )

#构建基础模型
base_model = InceptionV3(weights = 'imagenet', include_top = False)

#增加新的输出层
x = base_model.output
x = GlobalAveragePooling2D()(x) #将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dropout(0.5)(x)
#x = Dense(1024, activation = 'reul')(x)
predictions = Dense(nb_classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

#def setup_to_transfer_learning(model,base_model):
#    for layer in base_model.layers:
#        layer.trainable = False
#    model.compile(
#            optimizer = 'adam',
#            loss = 'categorical_crossentropy',#多分类，打印出loss
#            metrics=['accuracy'],          #打印出准确率
#            )
#    
def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17#前17层固定
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
        
    model.compile(
            optimizer = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9),
            loss = 'categorical_crossentropy',#多分类，打印出loss
            metrics=['accuracy'], 
            )

#全部固定的finetune
#setup_to_transfer_learning(model, base_model)
#history_tl = model.fit_generator(
#                    generator = train_generator,
#                    steps_per_epoch = len(train_generator)+1,
#                    epochs = nb_epoch,
#                    validation_data = val_generator,
#                    validation_steps = len(val_generator)+1,
#                    class_weight = 'auto',
#                    )

setup_to_fine_tune(model, base_model)

filepath = "weight-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')

history_ft = model.fit_generator(
                    generator = train_generator,
                    steps_per_epoch = len(train_generator)+1,
                    epochs = nb_epoch,
                    validation_data = val_generator,
                    validation_steps = len(val_generator)+1,
                    class_weight = 'auto',
                    )
model.save = ('keras_finrtune_model.h5')