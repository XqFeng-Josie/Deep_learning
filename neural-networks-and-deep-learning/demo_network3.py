#coding=utf-8
#CNN神经网络算法的测试

import network3
from network3 import Network
from network3 import ConvPoolLayer,FullyConnectedLayer,SoftmaxLayer


import theano.tensor as T
def ReLU(z):return T.maximum(0.0,z)
training_data , validation_data,test_data = network3.load_data_shared()
#expanded_training_data,_,_ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
mini_batch_size=10

'''传统神经网络：包含一个全连接层和一个softmax层'''
'''net = Network([
    FullyConnectedLayer(n_in=784,n_out=100),
    SoftmaxLayer(n_in=100,n_out=10)],mini_batch_size)
net.SGD(training_data,60,mini_batch_size,0.1,validation_data,test_data)'''
'''卷积神经网络：加入一个convolution层'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),filter_shape=(20,1,5,5),
                  poolsize=(2,2),activation_fn=ReLU),
    FullyConnectedLayer(n_in=20*12*12,n_out=100),
    SoftmaxLayer(n_in=100,n_out=10)],mini_batch_size)
net.SGD(training_data,10,mini_batch_size,0.1,validation_data,test_data)
'''卷积神经网络，包含两个convolution层,并且增大了训练集'''
'''net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),filter_shape=(20,1,5,5),
                  poolsize=(2,2),activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size,20,12,12),
                  filter_shape=(40,20,5,5),
                  poolsize=(2,2),activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40*4*4,n_out=1000,activation_fn=ReLU,p_dropout=0.5),
    SoftmaxLayer(n_in=1000,n_out=10,p_dropout=0.5)],mini_batch_size)
net.SGD(expanded_training_data,40,mini_batch_size,0.03,validation_data,test_data)'''
'''卷积神经网络，包含两个convolution层和两个fullyconnect层'''
'''net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),filter_shape=(20,1,5,5),
                  poolsize=(2,2),activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size,20,12,12),
                  filter_shape=(40,20,5,5),
                  poolsize=(2,2),activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40*4*4,n_out=1000,activation_fn=ReLU,p_dropout=0.5),
    SoftmaxLayer(n_in=1000,n_out=10,p_dropout=0.5)],mini_batch_size)
    FullyConnectedLayer(
        n_in=1000,n_out=1000,activation_fn=ReLU,p_dropout=0.5),
    SoftmaxLayer(n_in=1000,n_out=10,p_dropout=0.5)],mini_batch_size)
net.SGD(expanded_training_data,40,mini_batch_size,0.03,validation_data,test_data)'''
'''Ensemble of network：训练多个神经网络，投票决定结果，有时会提高'''