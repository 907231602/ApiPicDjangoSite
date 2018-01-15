#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-

import PicWeb.picHandle as picHandle
import PicWeb.my_one_hot as ones
import PicWeb.ResultAnalysis as analysisType
from PIL import Image
from PicWeb import *

#图片数组，图片名称
def predictBank(ar,name):

    # input image dimensions
    img_rows, img_cols = 200, 200
    # 训练的种类
    nb_classes = 9

    # the data, shuffled and split between  test sets
    (X_test, y_test) = picHandle.testPredictDataHandle200(ar)
    #(X_test, y_test) = picHandle.PredictDataHandle(ar)

    #print("shape===>>",X_test.shape[0])

    # 根据不同的backend定下不同的格式
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


    X_test = X_test.astype('float32')
    X_test /= 255
    #print(X_test.shape[0], 'test samples')
    #print("X_test===>>>>",X_test) #不要随意打印，打印就报错，很奇怪
    # 转换为one_hot类型
    #Y_test = ones.one_hot_ten(y_test, nb_classes)
    #print('one-hot-test:', Y_test) #不要随意打印，打印就报错，很奇怪


    result = picType_class(X_test) #model.predict(X_test)
    #print("加载模型")
    listOne = result[0:28]  #获取4*7=28张图片的结果

    oneResult = analysisType.resultType(listOne)


    # js_信用卡: 1 # js_投资 :  2  # js_生活':  3  # js_登录':  4  # js_网银':  5
    # js_贷款',  6 # js_资产',  7  # js_转账',  8  # js_首页',  9
    #print('oneResult=', oneResult)

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



