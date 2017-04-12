#coding=utf-8
#来自：深度学习进阶：算法与应用（麦子学院）
#神经网络算法：随机梯度下降的实现，实现手写数字的识别
import numpy as np
import random
class Network(object):
    #参数：sizes：list类型，代表每层神经元个数 net = Network([2,3,1])
    #np.random.randn(y,1):随机从正态分布（均值0，方差1）中生成
    #net.weights[1]存储连接第二层和第三层之间的权重

    def __init__(self,sizes):

        self.num_layers = len(sizes)#层数
        self.sizes = sizes
        #偏向初始化，除了输入层，每个神经元带有一个偏向
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        #权重初始化，每个连接都存在一个权重
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    # a` = η（wa + b） η是sigmod函数（01标准化） η= 1/1+e^(-x)
    #参数a为输入，向前的神经网络，返回输出
    def feedforword(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    #随机梯度下降算法（SGD）的实现
    #训练集整体是一个List类型。每个list包含很多个tuple(元组)：（x,y）.每个元组代表一个实例
    #参数：self，训练集，训练次数，块大小，学习率（η）,测试集
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data :
            n_test = len(test_data)
        n = len(training_data)
        #开始训练
        for j in xrange(epochs):
            #将训练集的数据随机打乱
            random.shuffle(training_data)
            #将数据集按块取出
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                #更新weight和bias
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}:{1}/{2}".format(
                    j,self.evaluate(test_data),n_test)
            else:
                print "Epoch {0} complete",format(j)
    #更新权重和偏向
    #参数：单个块数据，学习率
    #backpropagation算法：算出b和w的偏导。计算偏导的方法。
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            #目标函数的权重和偏向的误差(针对每个单独的实例)
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)

            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # w = old_w - eta * w~（b 同理）
        self.weights = [w-(eta/len(mini_batch))* nw
                        for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        #print self.weights[1:100]
    #返回值:目标函数的权重和偏向的误差
    #偏向的偏导 = 误差
    #权重的偏导 = 误差*权值
    # 参数:self,一个单独的实例
    def backprop(self, x, y):
        '''步骤：
                #输入x：设置输入层activaction a
                正向更新：对于l = 1,2,3...L,计算中间变量ｚ = wa+b，再sigmoid函数标准化
                计算输出层的误差error
                反向更新error
                '''
        # 初始化权重和偏向的矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        # 将activation列表化
        activations = [x]
        # 存储中间变量 Z = wa+b向量
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 计算输出层误差
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # 输出层更新，bias = err,weight = err . 倒数第二层的矩阵转置
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 从倒数第二层开始循环
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # 计算中间层误差
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    #每训练一轮，计算准确率
    def evaluate(self,test_data):
        test_reaults = [(np.argmax(self.feedforword(x)),y)
                        for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_reaults)
    #计算输出值与标签值的差
    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

    #对z进行标准化
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    #对实际值的sigmod值求导数运算
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


