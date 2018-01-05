#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-
import numpy as np
#每张图2940，训练集的one-hot
def to_one_hot(y,nb_class):
    len=y.shape[0]
    x=np.zeros(len*nb_class).reshape((len,nb_class))
    #print('y[0]=',len)
    #图片分三类，除以3可以把图片分为三个范围，第一个范围是第一张图片，以此类推
    type=len/nb_class
    for i in range(len):
        #print('i==',i)
        if(i < type ):
            x[i][0]=1
            continue
        elif (i < type * 2):
            x[i][1] = 1
            continue
        elif (i < type * 3):
            x[i][2] = 1
            continue
        elif (i < type * 4):
            x[i][3] = 1
            continue
        elif (i < type * 5):
            x[i][4] = 1
            continue
        elif (i < type * 6):
            x[i][5] = 1
            continue
        elif (i < type * 7):
            x[i][6] = 1
            continue
        elif (i < type * 8):
            x[i][7] = 1
            continue
        else:
            x[i][8]=1
    print('one-->',x)
    return x

#测试集的one-hot
def one_hot_ten(y,nb_class):
    len=y.shape[0]
    x = np.zeros(len* nb_class).reshape((len, nb_class))
    #print(y)
    #print('ten=',len)
    for j in range(len):
        if(y[j] == '1' ):
            x[j][0]=1
        elif(y[j] == '2' ):
            x[j][1]=1
        elif (y[j] == '3'):
            x[j][2] = 1
        elif (y[j] == '4'):
            x[j][3] = 1
        elif (y[j] == '5'):
            x[j][4] = 1
        elif (y[j] == '6'):
            x[j][5] = 1
        elif (y[j] == '7'):
            x[j][6] = 1
        elif (y[j] == '8'):
            x[j][7] = 1
        else:
            x[j][8]=1
    return x


if __name__ == "__main__":

    y=one_hot_ten(3,4)
    print(y)