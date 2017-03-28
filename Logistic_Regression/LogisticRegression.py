#coding=utf-8
#简单梯度下降实现logistic_regression
import numpy as np
import random
#梯度下降更新Θ值
#参数：矩阵（序列），向量（类别），最终学习的参数值Θ，学习率，实例数，循环次数（更新次数）
def gradientDescent(x,y,theta,alpha,m,numIterations):
    #矩阵转置
    xTrans = x.transpose()
    for i in range(0,numIterations):
        hypothesis = np.dot(x,theta)
        #预测和实际值之间的差
        loss = hypothesis - y
        #cost函数定义具有自由度，这里主要根据真实值和预测值之间的误差定义，cost逐渐减小
        cost = np.sum(loss **2)/ (2*m)
        print "Iteration %d | cost :%f"%(i,cost)
        gradient = np.dot(xTrans,loss) / m
        theta = theta - alpha * gradient;
    return theta
#创建数据进行测试训练
#参数：实例数，偏好，方差
def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=numPoints)#默认一列
    for i in range(0,numPoints):#返回{0....numPoints-1}
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i+bias) + random.uniform(0,1)*variance
    return x,y

x,y = genData(100,25,10)
print ("x:")
#print (x)
print ("y")
#print (y)
m,n = np.shape(x)
print m,n
m_y = np.shape(y)
print "y Length:",str(m_y)
#更新次数
numIterations = 100000
#学习效率（太大则可能跨过决定值，太小则学习效率低，因此会动态更新）
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x,y,theta,alpha,m,numIterations)
print theta  #打印出Θ0，Θ1
#然后可以根据这两个训练出来的参数进行预测
x_pred = 104
y_pred = theta[0] + theta[1] * x_pred
print "y_pred: "+str(y_pred)