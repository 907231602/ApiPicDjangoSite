#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-
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


def Cnn_bank_run(*kwargs):
    basePath='static\\imageTrain'+'\\%s\\%s' % (kwargs[0],kwargs[1])#日期文件夹+系统类别文件夹
    filesname=os.listdir(basePath)
    for files in filesname:
        pass

    # 全局变量
    batch_size = 18
    nb_classes = 9      #类别，即种类
    epochs = 2
    # input image dimensions
    img_rows, img_cols =200,200
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # the data, shuffled and split between train and tNewest sets
    (X_train, y_train) = picHandle.trainDataBankHandle200(kwargs[1],basePath) #系统名称+
    (X_test, y_test) = picHandle.testDataBankHandle200(kwargs[1],basePath)
    #print(len(X_train[0][0]))
    #print(len(X_train[0]))
    #print(X_train[0])
    #print('---->',X_train.shape)

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
    #print('X_train shape:', X_train.shape)
    #print(X_train.shape[0], 'train samples')
    #print(X_test.shape[0], 'test samples')

    # 转换为one_hot类型
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_train=ones.to_one_hot(y_train,nb_classes)
    #print('one-hot-train:',Y_train)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_test =ones.one_hot_ten(y_test,nb_classes)
    #print('one-hot-test:',Y_test)
    modelTrain.modelTrain_ByType(kwargs[1],X_train,Y_train,X_test,Y_test)
    #modelTrain_ByType(kwargs[1],X_train,Y_train,X_test,Y_test)

    '''
    # 构建模型
    model = Sequential()
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='same',
                            input_shape=input_shape))  # 卷积层1
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
    model.add(Dropout(0.25))  # 神经元随机失活
    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(128))  # 全连接层1
    model.add(Activation('relu'))  # 激活层
    model.add(Dropout(0.5))  # 随机失活
    model.add(Dense(nb_classes))  # 全连接层2
    model.add(Activation('softmax'))  # Softmax评分

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, Y_test))
    
    
    # 评估模型
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save('static\\h5\\%s.h5' % kwargs[1])

    result=model.predict(X_test)

    print('result=',result)
    '''




