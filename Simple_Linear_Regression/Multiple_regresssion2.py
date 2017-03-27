#coding=utf-8
#多元线性回归的应用(有分类型变量)
#送货问题（运输里程和运输次数以及分类型变量车型对总运输时间的影响 x1,x2,x3,x4,x5,b0,b1,b2,b3,b4,b5）
#数据集已经去掉了表头(data1.csv)
#将分类型的变量表示成历史那有规律的数据(0,1型)
# y_hat = b0 + b1*x1 + b2*x2 +...+ bp*xp
from numpy import genfromtxt #格式转换
import  numpy as np
from sklearn import datasets,linear_model

dataPath = r"F:\pycharmprojects\Simple_Linear_Regression\data1.csv"
deliveryData  = genfromtxt(dataPath,delimiter=(','))
print ("data:")
print (deliveryData)
#取出数据
X = deliveryData[:,:-1]
#取出类别标签
Y = deliveryData[:,-1]
print ("X:")
print (X)
print ("Y:")
print Y
#定义一个模型
regr = linear_model.LinearRegression()
regr.fit(X,Y)
#打印模型的参数预测，b1,b2，b3,b4,b5
print ("coefficients:")
print regr.coef_
#打印b0(截距)
print "intercept"
print regr.intercept_
xPred = [102,6,1,0,0]
yPred = regr.predict(xPred)
print ("predicted y:")
print yPred #10.90757981