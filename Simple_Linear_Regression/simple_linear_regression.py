#coding=utf-8
#简单线性回归问题的实现（汽车销售预测） y0 = b0 + b1 * x
import numpy as np
#传入参数集，返回b0,b1
def fitSLR (x,y):
     n = len(x)
     dinominator = 0 #分母
     numerator = 0 #分子
     for i in range(0,n):
         numerator += (x[i]-np.mean(x))*(y[i]-np.mean(y))
         dinominator += (x[i]-np.mean(x))**2
     print 'numerator:',numerator
     print 'dinominator:',dinominator
     b1 = numerator/float(dinominator)
     b0 = np.mean(y)-b1* np.mean(x)
     return b0,b1
def predict(x,b0,b1):
    return b0+b1*x
x = [1,3,2,1,3]#月份
y =[14,24,18,17,37]#销售数量
b0,b1 = fitSLR(x,y)
print 'b0:',b0
print 'b1:',b1
x_test = 6
y_test = predict(6,b0,b1)
print "y_test:",y_test