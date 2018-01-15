#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-
from django.http import JsonResponse
from PicWeb.PreDictBankFunc import predictBank
from PicWeb.models import Pic
import simplejson
from django.core.serializers import serialize
from django.test import override_settings
import numpy as np
from PIL import Image
import datetime
import os
from . import Cnn_Bank_train as bankTrain
from . import ScreenExpansion as screen

# Create your views here.

#图片分类、识别,返回json类型的图片数据
@override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=10242880)
def picAnalysis(request):
    try:
        req = simplejson.loads(request.body)
        name = req['Name']
        #print(name)
        # 对图片进行处理，把传过来的数组进行还原
        listpic = req['KeyWord']
        ar = np.array(listpic).reshape(1366,728)
        ar = ar.T  # 倒置,目的是把图片还原
        im = Image.fromarray(ar.astype('uint8'))
        #修改屏幕大小
        screen.ScreenExpans(im)
        prenum=predictBank(im, name)  #对图片进行预测，并返回预测结果
        im.close()
        list = ['信用卡', '投资', '生活', '登录', '网银', '贷款', '资产', '转账', '首页']

        pic = Pic()
        pic.picName = name
        pic.picNumType = prenum
        if (prenum == 0):
            pic.picTypeName='未知页面'
        else:
            pic.picTypeName = list[prenum - 1]
        d = simplejson.loads(serialize('json', [pic])[1:-1])
        return JsonResponse(d)
    except ConnectionResetError as ce:
        print("远程主机强迫关闭了一个现有的连接",ce)

# 根据系统名称进行分类图片分类、识别,返回json类型的图片数据
@override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=10242880)
def picAnalysisBySysName(request):
    try:
        req = simplejson.loads(request.body)
        name = req['Name']  #系统名称
        # 对图片进行处理，把传过来的数组进行还原
        listpic = req['KeyWord']  #图片数组
        print(np.array(listpic).shape)
        ar = np.array(listpic).reshape(1366, 728)
        ar = ar.T  # 倒置,目的是把图片还原
        im = Image.fromarray(ar.astype('uint8'))
        # im.save('static\\image\\tt.png')
        prenum = predictBank(im, name)  # 对图片进行预测，并返回预测结果
        # print('image-->', im.size)
        im.close()
        list = ['信用卡', '投资', '生活', '登录', '网银', '贷款', '资产', '转账', '首页']

        pic = Pic()
        pic.picName = name
        pic.picNumType = prenum
        #如果返回值为0,表示该页面不属于该系统页面
        print(type(prenum))
        if (prenum == 0):
            pic.picTypeName='未知页面'
        else:
            pic.picTypeName = list[prenum - 1]
        d = simplejson.loads(serialize('json', [pic])[1:-1])
        return JsonResponse(d)
    except ConnectionResetError as ce:
        print("远程主机强迫关闭了一个现有的连接", ce)



#保存图片，以日期(天)为单位，在日期文件下判断类别是否存在，否就创建，在类别下保留图片.  日期->类别->图片
#打算每周删除一些前一周的图片
def savePic(request):
    try:
        basepath='static\\imageTrain'
        req = simplejson.loads(request.body)
        name = str(req['Name']).split('_')

        # 对图片进行处理，把传过来的数组进行还原
        listpic = req['KeyWord']
        print('%s' % listpic)
        ar = np.array(listpic).reshape(2,4)
        ar = ar.T  # 倒置,目的是把图片还原
        im = Image.fromarray(ar.astype('uint8'))

        nowFile=datetime.datetime.now().strftime('%Y%m%d')  #今天文件
        now=datetime.datetime.now().strftime('%Y%m%d%H%M%S')#当前时间
        is_exit=os.path.exists(basepath+'\\%s' % nowFile)
        if(is_exit):
            pass
        else:
            os.makedirs(basepath+'\\%s' % nowFile)
        is_fileExit=os.path.exists(basepath+'\\%s\\%s' % (nowFile,name[0]))
        if (is_fileExit):
            pass
        else:
            os.makedirs(basepath+'\\%s\\%s' % (nowFile,name[0]))
        im.save(basepath+'\\%s\\%s\\%s_%s.png' % (nowFile,name[0],now,name[1]))

        return JsonResponse({'info':'Ok'})
    except BaseException as e:
        print('savePic Exception ======>>', e)
        return JsonResponse({'info': 'error'})


#图片训练,根据不同系统，训练成不同系统的库，并标记已经训练的库
def trainPic(request):
    basePath='static\\imageTrain'
    listFile = os.listdir(basePath)
    #标记并判断，如果以“_”结尾，则不处理
    for lineData in listFile:
        if lineData.endswith('_'):
            continue
        else:
            for linefile in os.listdir(basePath+'\\%s' % lineData):
                if linefile.endswith('_'):
                    continue
                else:
                    #bankTrain.getFile(lineData,linefile)
                    bankTrain.Cnn_bank_run(lineData,linefile)
                    os.renames(basePath+'\\%s\\%s' % (lineData,linefile),basePath+'\\%s\\%s%s' % (lineData,linefile,'_'))#标记训练完成
            os.renames(basePath+'\\%s' % lineData,basePath+'\\%s%s' % (lineData,'_'))                 #标记训练完成

    return JsonResponse({'num':listFile.__len__()})


def filepath():
    print(os.path.exists('static\\imageTrain'))



if __name__=='__main__':
    filepath()
