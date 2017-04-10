#coding=utf-8
#扩展数据集：将数据的像素从上下左右四个方位移动一个像素位置，生成25000个数据

from __future__ import print_function
import cPickle
import gzip
import os.path
import random

import numpy as np
print ("Expanding the MNIST training set")

if os.path.exists("data/mnist_expanded.pkl.gz"):
    print ("The expanned training set already exists. Exiting.")
else:
    f = gzip.open("data/mnist.pkl.gz","rb")
    training_data,validation_data,test_data = cPickle.load(f)
    f.close()
    expanded_training_paris = []
    j = 0
    for x, y in zip(training_data[0],training_data[1]):
        expanded_training_paris.append((x,y))
        image = np.reshape(x,(-1,28))#将数据集x转化成28列，行数计算得知
        j+=1
        if j % 1000 == 0:
            print ("Expanding image number",j)
        #numpy.roll(x,d,axis):axis=0代表上下，axis=1代表左右，d>0代表上右，d<0代表左下
        for d,axis,index_position,index in[
            (1,0,"first",0),
            (-1,0,"first",27),
            (1,1,"last",0),
            (-1,1,"last",27)]:
            new_img = np.roll(image,d,axis)
            if index_position == "first":
                new_img[index:]=np.zeros(28)#取出第index行所有列
            else:
                new_img[:index] = np.zeros(28)
            expanded_training_paris.append((np.reshape(new_img,784),y))#扁平化
    random.shuffle(expanded_training_paris)#打算调整过后的数据集
    expanded_training_data = [list(d) for d in zip(*expanded_training_paris)]
    print ("Saving expand data.This may take few minutes.")
    f = gzip.open("data/mnist_expanded.pkl.gz","w")
    cPickle.dump((expanded_training_data,validation_data,test_data),f)
    f.close()