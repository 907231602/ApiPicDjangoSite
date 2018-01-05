from PIL import Image
import numpy as np
import math

#灰度处理,并转化成数组
def imageHanle(imgName):
    imageHandle = Image.open(imgName)
    L= imageHandle.convert('L')   #转化为灰度图
    im_array = np.array(L)
    return im_array


#传入一张图片，并对图片进行切割，把切割好的图片转化数组加入到list
def testPredictDataHandle200(im):
    X = list()
    Y = list()
    im_size = im.size
    print("图片信息：",im_size)

    numX=im_size[0]/200
    numY=im_size[1]/200
    # 第1块           个数
    w = im_size[0] / numX  # 设置被切长度  13.66/2 的倍数
    h = im_size[1] / numY  # 设置被切宽度  7.28/2的倍数
    print(w,h)
    x = 0  # 长
    y = 0  # 宽

    for i in range(math.ceil(numY)):  # 循环宽度4次
        for j in range(math.ceil(numX)):  # 循环长度7次
            region = im.crop((x, y, x + w, y + h))
            X.append(np.array(region))      #将图片切割，并把切割好的图片数组保存
            Y.append(1)
            #region.save("static\\imageTests\\crop_average_8-%d-%d-%d.png" % (k, i, j))
            x = x + w
            y = y
        x = 0  # 高依次增加，宽度从0~~边界值
        y = y + h

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)


if __name__ == "__main__":
    path='..\\static\\image\\生活_t.png'
    imageHandle = Image.open(path)
    L = imageHandle.convert('L')  # 转化为灰度图
    im_array = np.array(L)
    x,y=testPredictDataHandle200(im_array)
    #for i in range(28):
    print(x.shape)

