#coding=utf-8
'''#SVM支持向量机算法的实现
#使用sklearn交际的SVM库实现
from sklearn import svm
x = [[2,0],[1,1],[2,3]]#实例
y = [0,0,1]#类别
#建立分类器模型
clf = svm.SVC(kernel='linear')
clf.fit(x,y)
print clf
#支持向量点
print clf.support_vectors_
#支持向量点的索引
print clf.support_
#对于每个类别找到的支持向量个数
print clf.n_support_
#预测一点的类别
print clf.predict([2,0])'''
#解决大一点的数据集
print (__doc__)
import numpy as np
import pylab as pl
from sklearn import svm
#创建40个点
np.random.seed(0)#使得每次运行程序时产生点一致
#生成线性可区分的两类点
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
#生成各自的类别标记
Y = [0]*20+[1]*20
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

#画出超平面
#w0*x+w1*y+w3=0换成点斜式  y = -(w0 / w1)x - (w3 / w1)

w = clf.coef_[0]
a = -w[0]/w[1]#斜率
#从-5到5产生一些x的值
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0] / w[1])
#画出支持向量所在线，即三条线平行，截距不同
#取出第一个支持向量，并算出截距，并算出下面的那条线
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1]-a * b[0])
#取出倒数第一个支持向量，并算出截距，并算出上面的那条线
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1]-a * b[0])
#画图
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolor='none')
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()