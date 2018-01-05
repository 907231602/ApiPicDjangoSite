#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-

from keras.models import Sequential
import numpy as np
import PicWeb.picHandle as picHandle
from keras import backend as K
from keras.models import load_model
import PicWeb.my_one_hot as ones
import PicWeb.ResultAnalysis as analysisType
from PIL import Image

#图片数组，图片名称
def predictBank(ar,name):
    # input image dimensions
    img_rows, img_cols = 200, 200
    # 训练的种类
    nb_classes = 9

    # the data, shuffled and split between  test sets
    (X_test, y_test) = picHandle.testPredictDataHandle200(ar)

    print("shape===>>",X_test.shape[0])

    # 根据不同的backend定下不同的格式
    if K.image_dim_ordering() == 'th':
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        # input_shape = (1, img_rows, img_cols)
    else:
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        #input_shape = (img_rows, img_cols, 1)

    X_test = X_test.astype('float32')
    X_test /= 255
    print(X_test.shape[0], 'test samples')
    print("X_test===>>>>",X_test)
    # 转换为one_hot类型
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_test = ones.one_hot_ten(y_test, nb_classes)
    print('one-hot-test:', Y_test)

    # 构建模型
    model = Sequential()

    model = load_model('E:\PyCharmWork\CNN_Bank\CnnBankUp.h5')
    result = model.predict(X_test)
    print("加载模型")
    listOne = result[0:28]  #获取4*7=28张图片的结果

    oneResult = analysisType.resultType(listOne)

    # js_信用卡: 1 # js_投资 :  2  # js_生活':  3  # js_登录':  4  # js_网银':  5
    # js_贷款',  6 # js_资产',  7  # js_转账',  8  # js_首页',  9
    print('oneResult=', oneResult)

    return oneResult




if __name__ == "__main__":

    #picOpen('..\\static\\image\\生活_t.png')
    path = '..\\static\\image\\生活_t.png'
    imageHandle = Image.open(path)
    L = imageHandle.convert('L')  # 转化为灰度图
    im_array = np.array(L)
    #np.savetxt('a.txt',im_array)
    #print(im_array)
    im = Image.fromarray(im_array.astype('uint8'))  #对图片进行复原
    predictBank(im,'pic')
    predictBank(im, 'pic2')



