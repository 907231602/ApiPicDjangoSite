from keras.models import load_model
import numpy as np

#参考：https://zhuanlan.zhihu.com/p/27101000

print('load model...')
model = load_model('static\\CnnBankUp.h5', compile=False)
print('load done.')


print('test model...')
model.predict(np.zeros((2, 200,200,1)))
#print(model.predict(np.zeros((2, 200,200,1))))
print('test done.')


# 使用模型，在得到用户输入时会调用以下两个函数进行实时文本分类
# 输入参数 comment 为经过了分词与向量化处理后的模型输入
def picType_class(comment):
    global model
    result_vec = model.predict(comment)
    return result_vec
