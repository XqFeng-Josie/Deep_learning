#coding=utf-8
#回归中相关度r和R平方值的应用
#如果是简单线性回归，R平方值 = sqrt（r）
#如果是多元的，则格外讨论
import numpy as np
from astropy.units import Ybarn
import math
#计算相关度
def computeCorrelation(X,Y):
    xBar = np.mean(X)#均值
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY =0
    for i in range(0,len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2
    SST = math.sqrt(varX * varY)
    return SSR/SST
#多元情况
#参数：自变量，因变量，方程的最高幂次
def polyfit(x,y,degree):
    result = {}
    coeffs =np.polyfit(x,y,degree)
    #算出各类参数b0，b1...
    result['polynomial'] = coeffs.tolist()
    #返回预测值
    p = np.poly1d(coeffs)
   # R^2 = ssr/sst = ∑(y_hat-y_avg)^2/∑(y-y_avg)^2
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    print "ssreg:",str(ssreg)
    print "sstot:", str(sstot)
    result['determination'] = ssreg / sstot
    #包含斜率和截距b0,b1以及决定系数rr
    print "result:",result
    return result

testX = [1,3,8,7,9]
testY = [10,12,24,21,34]
print "r="+str(computeCorrelation(testX,testY))
#是简单回归的情况下，决定系数计算
print "r^2="+str((computeCorrelation(testX,testY))**2)
#多元情况下决定系数的计算
print polyfit(testX,testY,1)['determination']


