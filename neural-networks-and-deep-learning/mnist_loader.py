#coding=utf-8
#来自：深度学习进阶：算法与应用（麦子学院）
#下载数据集的函数
#下载的数据集包含训练数据（50，000），验证数据（10，000）以及测试数据（10,000）
#每张手写数字图片大小为28*28-784

import cPickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
#基于上述load_data()函数，不过下载的数据集更好的格式化，适合我们的神经网络结构调用。
#数据是以列表的形式返回，列表里每个元素为一个元组（x,y）,x是一个（784，1）的矩阵，代表图片的像素点
#y 是（10,1）的矩阵，代表最终的图片数字类别
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
