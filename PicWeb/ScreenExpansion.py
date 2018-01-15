#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-
#扩展图片像素：把截到的屏幕分辨率扩展到1920*1080

import numpy as np
from PIL import Image

#填补多出像素，用0填补,返回1920*1080的数组
def ScreenExpans(screen):
    arr1 = np.zeros((1080,1920))  # 没有颜色的就为0
    screen = np.array(screen)
    tt = screen.shape

    y = tt[0]
    x = tt[1]
    #如果屏幕超出范围，只赋予截到的屏幕1920*1080的值
    if(x>1920):
        x=1920
    if(y>1080):
        y=1080

    for k in range(x):
        for t in range(y):
            arr1[t][k] = screen[t][k]
    #im=Image.fromarray(arr1.astype('uint8'))
    #im.save('ss.png')
    return Image.fromarray(arr1.astype('uint8'))


if __name__ == '__main__':

    path = '..\\static\\image\\生活_t.png'
    imageHandle = Image.open(path)
    L = imageHandle.convert('L')  # 转化为灰度图
    im_array = np.array(L)
    ScreenExpans(im_array)
    # im=Image.fromarray(sc.astype('uint8'))
    # im.save('ps.png')

