#coding=utf-8
#实现提高版的Mnist手写数字识别
#不同之处在于：1.权重的初始化由从标准正态分布生成，变成由均值为0，标准差为1/sqrt(n_in),n_in为输入神经元个数
                #2.Cost函数：cross-entropy
                #3.Regularization：L1,L2
                #4.Softmax layer  替换s igmoid layer

import numpy as np
import json
import random
import sys
#传统的二次cost
class QuadraticCost(object):
    @staticmethod
    #对于一个x,y和单个的神经元 求cost C = (a-y)^2/2
    def fn(a,y):
        return 0.5 * np.linalg.norm(a-y)**2
    @staticmethod
    #对于输出层求误差delta 即w的偏导数=（a - y）* （z 的偏导数）* x
    def delta(z,a,y):
        return (a-y) * sigmoid_prime(z)
# 新的cost函数
class CrossEntropyCost(object):
    @staticmethod
    # 对于一个x,y和单个的神经元 求cost C = -y.lna-(1-y)ln(1-a)
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    # 对于输出层求误差delta，即w的偏导数=（a - y）* x
    def delta(z, a, y):
        return (a - y)

#神经网络结构类,默认的cost函数是新的cost-entropy函数
class Network(object):
    def __init__(self,sizes,cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #设置默认的初始化权重的方法
        self.default_weight_initializer()
        self.cost = cost
#权重的初始化生成 由均值为0，标准差为1/sqrt(n_in)的正态分布随机生成,n_in为输入神经元个数
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x)
                        for x,y in zip(self.sizes[:-1],self.sizes[1:])]
#原始初始化权重的方法，即在标准正态分布上随机生成
    def large_weight_initializer(self):
        self.biases  = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x)
                        for x,y in zip(self.sizes[:-1],self.sizes[1:])]
# a` = η（wa + b） η是sigmod函数（01标准化） η= 1/1+e^(-x)
# 参数a为输入，向前的神经网络，返回输出
    def feedforword(self,a):
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
#随机梯度下降算法（SGD）的实现
#训练集整体是一个List类型。每个list包含很多个tuple(元组)：（x,y）.每个元组代表一个实例
#参数：self，训练集，训练次数，块大小，学习率（η）,regularization参数
    def SGD(self,training_data,epochs,mini_batch_size,eta,
            lmbda=0.0,
            evaluation_data = None,#设置评估数据的数据集
            monitor_evaluation_cost=False,#打印验证集cost
            monitor_evaluation_accuracy=False,#打印验证集的accuracy
            monitor_training_cost=False,#打印训练集的cost
            monitor_training_accuracy=False):#打印训练集的accuracy
        if evaluation_data:
            n_data  = len(evaluation_data)
        n = len(training_data)
        evaluation_cost,evaluation_accuracy = [],[]
        training_cost,training_accuracy = [],[]
        for j in xrange(epochs):
            random.shuffle(training_data)#打乱训练集
            # 将数据集按块取出
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0,n,mini_batch_size)]
            # 更新weight和bias
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda,len(training_data))
            print "Epoch %s training complete"% j
            if monitor_training_cost:
                cost = self.total_cost(training_data,lmbda)
                training_cost.append(cost)
                print "Cost on training date:{}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data,convert = True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data :{}/{}".format(accuracy,n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data,lmbda,convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data :{}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data :{}/{}".format(accuracy,n_data)
        return evaluation_cost,evaluation_accuracy,\
                training_cost,training_accuracy
        # 更新权重和偏向
        # 参数：单个块数据，学习率,regularization参数，训练集数据长度
        # backpropagation算法：算出b和w的偏导。计算偏导的方法。
    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_b = [np.zeros(b.shape)for b in self.biases]
        nabla_w = [np.zeros(w.shape)for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        # w = (1-eta*lmbda/n)*w - (eta/m) *w~
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
    # 返回值:目标函数的权重和偏向的偏导数
    # 偏向的偏导 = 误差
    # 权重的偏导 = 误差*权值
    # 参数:self,一个单独的实例
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforword
        activation = x
        activations = [x]
        zs=[]
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        # 计算输出层误差
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # 输出层更新，bias = err,weight = err . 倒数第二层的矩阵转置
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        for l in xrange(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)
    #计算准确率
    #convert=True.代表计算训练集的准确率，反之则是计算验证集的准确率
    #具体的处理差别参见mnist_loader.load_data_wrapper
    def accuracy(self,data,convert=False):
        if convert:
            results = [(np.argmax(self.feedforword(x)),np.argmax(y))
                       for (x,y)in data]
        else:
            results = [(np.argmax(self.feedforword(x)),y)
                       for (x,y) in data]
        return sum(int(x==y)for (x,y) in results)
    #计算cost
    def total_cost(self,data,lmbda,convert=False):
        cost = 0.0
        for x,y in data:
            a=self.feedforword(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        #L2 Rerularization C = C0+lmbda/2*n *(权重的平方和)
        cost += 0.5*(lmbda/len(data))* sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    #将学习好的神经网络保存
    def save(self,filename):
        data = {"sizes":self.sizes,
                "weights":[w.tolist() for w in self.weights],
                "biases":[b.toliat() for b in self.biases],
                "cost":str(self.cost.__name__)}
        f = open(filename,"w")
        json.dump(data,f)
        f.close()
#加载神经网络的参数数据文件
def load(filename):
    f = open(filename,"r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__],data["cost"])
    net = Network(data["sizes"],cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
#将0-9的数字向量化
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))





