#!/usr/bin/env python3.5.2
# -*- coding: utf-8 -*-

#结果分析：
def resultType(result):
    #print('total=', len(result))

    nb_type = 9  # 辨识图片种类
    num = len(result) / nb_type
    count = 0
    indexNum = list()
    for kk in range(len(result)):
        if(result[count][result[count].argmax()]>=0.5):
            index = result[count].argmax() + 1
            indexNum.append(index)
        count += 1

    list0_28 = indexNum[0:28]
    index1 = list0_28.count(1)
    index2 = list0_28.count(2)
    index3 = list0_28.count(3)
    index4 = list0_28.count(4)
    index5 = list0_28.count(5)
    index6 = list0_28.count(6)
    index7 = list0_28.count(7)
    index8 = list0_28.count(8)
    index9 = list0_28.count(9)
    #print('最大值=',max([index1,index2,index3]))
    maxValue=max([index1,index2,index3,index4,index5,index6,index7,index8,index9])
    if(maxValue>=len(result)/2-1):
        print('max=',maxValue,' ',len(result)/2-1)
    else:
        return 0
    if(index1 == maxValue):
        return 1
    elif(index2 == maxValue):
        return 2
    elif (index3 == maxValue):
        return 3
    elif (index4 == maxValue):
        return 4
    elif (index5 == maxValue):
        return 5
    elif (index6 == maxValue):
        return 6
    elif (index7 == maxValue):
        return 7
    elif (index8 == maxValue):
        return 8
    else:
        return 9


