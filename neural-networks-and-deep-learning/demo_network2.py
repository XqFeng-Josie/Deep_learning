#coding=utf-8
#测试network2的模块
import network2
import mnist_loader

training_data,validation_data,test_data = mnist_loader.load_data_wrapper()

print ("training data")
print(type(training_data))
print (len(training_data))
print (training_data[0][0].shape)
print (training_data[0][1].shape)
print ("validation data")
print(type(validation_data))
print (len(validation_data))
print ("test data")
print(type(test_data))
print (len(test_data))
#设定了专门的cost函数
net = network2.Network([784,30,10],cost=network2.CrossEntropyCost)
#与network初始化一致
#net.large_weight_initializer()
#调用随机梯度下降算法开始训练
net.SGD(training_data,30,10,0.5,evaluation_data=test_data,lmbda=5.0,
        monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,
        monitor_training_accuracy=True,monitor_training_cost=True)
