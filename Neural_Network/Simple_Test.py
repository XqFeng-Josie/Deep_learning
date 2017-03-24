#coding=utf-8
# #XOR运算集的实现
from Neural_Network import Neural_Network
import numpy as np
#设置一个输入层2个单元，隐藏层2个单元，输出层1个单元（layers）
nn  = Neural_Network([2,2,1],'tanh')
xx = np.array([[0,0],[0,1],[1,0],[1,1]])
yy = np.array([0,1,1,0])
nn.fit(xx,yy)
for i in [[0,0],[0,1],[1,0],[1,1]]:
     print (i,nn.predict(i))
     