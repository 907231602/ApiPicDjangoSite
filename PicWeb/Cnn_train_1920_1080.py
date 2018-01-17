#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-

#训练图片，图片大小（1920*1080）


from . import my_one_hot as ones
import numpy as np
np.random.seed(1520)  # for reproducibility
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D

from . import picHandle
from PicWeb import trainModel_init as modelTrain
from keras import backend as K
import os

#http://blog.csdn.net/lujiandong1/article/details/55806435

def getFile(*kwargs):
    print(kwargs)


def Cnn_run(*kwargs):
    basePath='static\\imageTrain'+'\\%s\\%s' % (kwargs[0],kwargs[1])#日期文件夹+系统类别文件夹
    filesname=os.listdir(basePath)
    listType=list()
    for files in filesname:
        listType.append(files[15:].split('.')[0])

    nb_classes=set(listType).__len__()

    # 全局变量
    img_rows, img_cols =200,200

    # the data, shuffled and split between train and tNewest sets
    (X_train, y_train,Y_train) = picHandle.trainDataBankHandle200(kwargs[1],basePath) #系统名称+路径
    (X_test, y_test,Y_test) = picHandle.testDataBankHandle200(kwargs[1],basePath)

    # 根据不同的backend定下不同的格式
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 转换为one_hot类型
   # Y_train=ones.to_one_hot(y_train,nb_classes)
    #Y_test =ones.one_hot_ten(y_test,nb_classes)

    modelTrain.modelTrain_ByType(kwargs[1],X_train,Y_train,X_test,Y_test)





