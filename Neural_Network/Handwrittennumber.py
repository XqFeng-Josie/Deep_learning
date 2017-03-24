#coding=utf-8
#手写数字的识别
#每个图片是8*8
#识别数字0.1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from  sklearn.preprocessing import LabelBinarizer#将图片转为二维数字类型
from Neural_Network import Neural_Network
from sklearn.cross_validation import train_test_split#拆分数据集

digits  = load_digits()
X = digits.data#特征量
y = digits.target#标签（0——9）
#将所有的值转化到0-1之间
X -= X.min()
X /= X.max()
#64个像素点，区分十个数字
nn = Neural_Network([64,100,10],'logistic')
X_train,X_test,y_train,y_test = train_test_split(X,y)
#将标记转化为0，1的数，如3转为00010000000，如5转化为0000010000
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print "start_fitting"
nn.fit(X_train,labels_train,epochs=3000)
#装入预测值
predictions=[]
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    #选出最大概率的数的位置即为预测值
    predictions.append(np.argmax(o))
#建立矩阵，对角线上的数越大，准确率越高
print confusion_matrix(y_test,predictions)
#算出准确率
print classification_report(y_test,predictions)