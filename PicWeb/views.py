from django.http import HttpResponse,JsonResponse
from PicWeb.PreDictBankFunc import predictBank
from PicWeb.models import Pic
#from django.views.decorators.csrf import csrf_exempt
import simplejson
from django.core.serializers import serialize
#from django.test import SimpleTestCase, override_settings
from django.test import override_settings
import numpy as np
from PIL import Image

# Create your views here.

@override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=10242880)
def picAnalysis(request):
    req = simplejson.loads(request.body)
    name = req['Name']
    print(name)
    # 对图片进行处理，把传过来的数组进行还原
    listpic = req['KeyWord']

    #aa=picType_class(np.zeros((3, 200, 200, 1)))
    #print('aa===>',aa)

    ar = np.array(listpic).reshape(1366,728)
    ar = ar.T  # 倒置,目的是把图片还原
    im = Image.fromarray(ar.astype('uint8'))
    #im.save('static\\image\\tt.png')
    prenum=predictBank(im, name)  #对图片进行预测，并返回预测结果
    print('image-->', im.size)
    im.close()
    list = ['信用卡', '投资', '生活', '登录', '网银', '贷款', '资产', '转账', '首页']

    pic = Pic()
    pic.picName = name
    pic.picNumType = 2
    pic.picTypeName = list[prenum - 1]
    d = simplejson.loads(serialize('json', [pic])[1:-1])
    return JsonResponse(d)



