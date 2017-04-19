#coding=utf-8
#CNN 算法实现
#调用ＧＰＵ进行运算（Theano）
# Theano:声明shared variables,从而可以用ＧＰＵ计算
# Theano:方程实现了各类基本的数学运算
# Theano
#
import cPickle
import gzip
import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d


def Linear(z):return z #替代sigmoid
def ReLU(z): return T.maximum(0.0,z)#解决vanishing gradient问题->ReLU解决
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
#使用GPU计算
GPU = True
if GPU:
    print ("Trying to run under GPU. If this is not desired, then modify"+\
           "network3.py \n to set the GPU flag to False.")
    try:theano.config.device = 'gpu'
    except:pass  #it's already set.
    theano.config.floatX = 'float32'
else:
    print ("Runninh with a CPU .If this is not desired,then modify"+\
           "network3.py to set\n the GPU flag to True.")

#加载MNIST数据集
def load_data_shared(filename="data/mnist.pkl.gz"):
    f = gzip.open(filename,'rb')
    training_data,validation_data,test_data = cPickle.load(f)
    f.close()
    def shared(data):
        '''将数据存储至ＧＰＵ支持的shared 格式下,borrow=True即设置为共享'''
        shared_x = theano.shared(np.asarray(data[0],dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(np.asarray(data[1],dtype=theano.config.floatX),borrow=True)
        return shared_x,T.cast(shared_y,"int32")#将y标签设置为整型（0-9）
    return [shared(training_data),shared(validation_data),shared(test_data)]
#构造和训练神经网络的类
class Network(object):
    #layers：包含整个神经网络的结构
    def __init__(self,layers,mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        #将每层的的参数打包成一个list
        self.params = [params for layers in self.layers for params in layers.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        init_layer = self.layers[0]
        #传入两个self.x，是用于不用dropout的区别
        init_layer.set_inpt(self.x,self.x,self.mini_batch_size)
        for j in xrange(1,len(self.layers)):
            prev_layer,layer = self.layers[j-1],self.layers[j]
            layer.set_inpt(prev_layer.output,prev_layer.output_dropout,self.mini_batch_size)
        #
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
    #训练随机梯度下降网络 Training the neteork using mini_batch stochastic gradient descent
    #参数：self，训练集，轮数，mini_batch_size ,学习率，验证集，测试集，regularization参数
    def SGD(self,training_data,epochs,mini_batch_size,eta,validation_data,test_data,lmbda=0.0):
        training_x,training_y = training_data
        validation_x,validation_y = validation_data
        test_x,test_y = test_data
        #计算各个数据集的mini数目
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        #用symbolic valiable计算
        #计算所有层的权重平方和
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        #定义cost function ，用了L2 regularization C=C0+ （lmbda/2*m）*权重的平方和
        cost = self.layers[-1].cost(self)+ 0.5*lmbda*l2_norm_squared/num_training_batches
        #对cost以及对应的参数求偏导，并更新
        grads = T.grad(cost,self.params)
        updates = [(param,param-eta*grad) for param , grad in zip(self.params,grads)]
        #对每个mini_batch训练,并计算准确率
        #i表示mini_batch的索引
        i = T.lscalar()
        train_mb = theano.function(
            [i],cost,updates=updates,
            givens={
               self.x:
               training_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
               self.y:
               training_y[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i],self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i * self.mini_batch_size:(i + 1) * self.mini_batch_size],
                self.y:
                    test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i],self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[i * self.mini_batch_size:(i+1)*self.mini_batch_size]
            })
        #开始实际的训练
        #设置最好的准确率
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for mini_batch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch +mini_batch_index
                if iteration % 1000 == 0:
                    print ("Training mini_batch number {0}".format(iteration))
                cost_ij = train_mb(mini_batch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j)
                                                   for j in xrange(num_validation_batches)])
                    print ("Epoch {0}:validation_accuracy {1:.2%}".format(epoch,validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print ("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j)
                                                     for j in xrange(num_test_batches)])
                            print ("The corresponding test accuracy is {0:.2%}".format(test_accuracy))
        print ("Finished training network.")
        print ("Best validation accuracy of {0:2%} obtained at iteration {1}"
               .format(best_validation_accuracy,best_iteration))
        print ("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
#卷积层，包含特征提取和特征映射
#定义卷积运算（convolutional）和池化运算(max_pooling)的类
class ConvPoolLayer(object):
    #参数：self，卷积图谱，图像大小，池化图谱，激活函数
    #filter_shape是一个长度为4 的元组（number of filters，number of imput features,the filter height,the filter width）
    #image_shape是一个长度为4 的元组（mini_batch_size,input feature maps,image_height,image_width）
    def __init__(self,filter_shape,image_shape,poolsize=(2,2),
                 activation_fn = sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        #初始化权重和偏向
        #eg:1*24*24/2*2=12*12=144个神经元（输出层）
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        #产生权值和偏向（平均值为0，方差为1/n_out产生）
        self.w = theano.shared(np.asarray(
                        np.random.normal(loc=0,scale=np.sqrt(1.0/n_out),size=filter_shape),
                        dtype=theano.config.floatX),borrow=True)
        self.b = theano.shared(np.asarray(
                        np.random.normal(loc=0,scale=np.sqrt(1.0/n_out),size=(filter_shape[0],)),
                        dtype=theano.config.floatX),borrow=True)
        self.params = [self.w,self.b]
    def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
                input=self.inpt,filters=self.w,filter_shape=self.filter_shape,
                image_shape=self.image_shape)
        pooled_out = pool_2d(
                input=conv_out,ws=self.poolsize,ignore_border=True)
        self.output =  self.activation_fn(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        self.output_dropout = self.output #在convolutional层没有dropout

#全连接层类
class FullyConnectedLayer(object):
    #构造函数，参数：self,n_in，n_out,激活函数，dropout大小
    def __init__(self,n_in,n_out,activation_fn=sigmoid,p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        #产生权重矩阵(shared变量类型)，正态分布产生
        self.w = theano.shared(np.asarray(np.random.normal(
                                loc=0.0,scale=np.sqrt(1.0/n_out),
                                size=(n_in,n_out)),
                                dtype=theano.config.floatX),#声明变量类型
                                name='w',borrow = True)
        #产生偏向向量
        self.b = theano.shared(np.asarray(np.random.normal(
                                loc=0.0,scale=np.sqrt(1.0/n_out),
                                size=(n_out,)),dtype=theano.config.floatX),
                                name='b',borrow = True)
        self.params = [self.w,self.b]
        #正向更新网络,两个版本，用dropout不用dropput
    def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size,self.n_in))
        #对保留的神经元进行运算
        self.output = self.activation_fn((1-self.p_dropout) * T.dot(self.inpt,self.w)+self.b)
        #找出最大的output
        self.y_out = T.argmax(self.output,axis=1)
        #......
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size,self.n_in)),self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout,self.w) + self.b)
        #计算准确率
    def accuracy(self,y):
        return T.mean(T.eq(y,self.y_out))
class SoftmaxLayer(object):
    def __init__(self,n_in,n_out,p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        #初始化权重和偏向
        self.w = theano.shared(
            np.zeros((n_in,n_out),dtype=theano.config.floatX),
            name='w',borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,),dtype=theano.config.floatX),
            name='b',borrow=True)
        self.params = [self.w,self.b]
    def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size,self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt,self.w)+self.b)
        self.y_out = T.argmax(self.output,axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size,self.n_in)),self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout,self.w)+self.b)
    def cost(self,net):
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]),net.y])
    def accuracy(self,y):
        return T.mean(T.eq(y,self.y_out))
def size(data):
    return data[0].get_value(borrow=True).shape[0]
def dropout_layer(layer,p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)
