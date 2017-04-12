#coding=utf-8
#实现一个神经网络算法
print __doc__
import numpy as np
def tanh(x):
    return np.tanh(x)
#tanh对应的导数
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)
def logistic(x):
    return 1/(1 + np.exp(-x))
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
class Neural_Network:

#构造函数:self:this  layers:list：层数以及每层神经元结点数，activation:模式（提供选择，默认为tanh）
    def __init__(self,layers,activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.weights=[]
        #从第一层到倒数第二层对权重随机赋值初始化
        for i in range(1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i+1] )) - 1) * 0.25)

# X是数据集，Y为类别标记，对网络的更新运算量很大，
# 不应该使用所有的实例来进行更新，而是每次抽取一定的样本。
# epochs：循环次数
    def fit(self,X,y,learning_rate=0.2,epochs=10000 ):
        #确定为最少2维的，然后将数据集转化成numpy矩阵
        X = np.atleast_2d(X)
        #初始化一个和X行数相同列数加一的矩阵，变量存1，便于对预测值进行赋值
        temp = np.ones([X.shape[0],X.shape[1]+1])
        #取所有行，取第一列到倒数第二列,进行数据集的赋值
        temp[:,0:-1]=X
        #x为矩阵类型且加入了预测值列
        X = temp
        # list=>array
        y = np.array(y)

        for k in range(epochs):
            #随机抽取一行实例对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a =[X[i]]
            #正向更新,得到每一层的值
            for l in range(len(self.weights)):

                a.append(self.activation(np.dot(a[l],self.weights[l])))
            # 对于输出层，得到运算值和真实值之间的差值（Tj-Oj）
            error = y[i] - a[-1]
            #对于输出层得到误差 Err =Oj(1-Oj)(Tj-Oj)
            deltas = [error * self.activation_deriv(a[-1])]

            #反向计算误差
            #从最后一层开始到第0层，每次退回一个
            for l in range(len(a)-2,0,-1):
                # Err = Oj(1-Oj)*Errk.wijk(点是做内积)
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            #权重更新 wij = wij + learning_rate * Errj.Oj
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i]+= learning_rate * layer.T.dot(delta)
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a =temp
            #预测最后的值
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
