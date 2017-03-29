#coding=utf-8
#简单聚类方法K-means方法的实现

import numpy as np
#数据集需要加一列
#参数：数据集，分为几类，迭代次数
def kmeans(X,k,maxIt):
    #返回行列维度
    numPoints,numDim = X.shape
    #增加一列作为分类标记
    dataSet = np.zeros((numPoints,numDim+1))
    # 所有行，除了最后一列
    dataSet[:,:-1] = X
    #随机选取K行，包含所有列
    centroids = dataSet[np.random.randint(numPoints,size = k),:]
    #centroids = dataSet[0:2,:]
    #给中心点分类进行初始化
    centroids[:,-1] = range(1,k+1)
    iterations = 0
    oldCentroids = None

    while not shouldStop(oldCentroids,centroids,iterations,maxIt):
        print "iteration:\n",iterations
        print "dataSet:\n", dataSet
        print "centroids:\n", centroids
        #不能直接用等号，不然会指向同一个变量
        oldCentroids = np.copy(centroids)
        iterations += 1
        #根据数据集以及中心点对数据集的点进行归类
        updateLabels(dataSet,centroids)
        # 更新中心点
        centroids = getCentroids(dataSet,k)
    return dataSet
#实现函数循环结束的判断
#当循环次数达到最大值，或者中心点不变化就停止
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations > maxIt:
        return True
    return  np.array_equal(oldCentroids,centroids)
 #根据数据集以及中心点对数据集的点进行归类
def updateLabels(dataSet,centroids):
    #返回行（点数），列
    numPoints,numDim = dataSet.shape
    for i in range(0,numPoints):
        #对每一行最后一列进行归类
        dataSet[i,-1] = getLabelFromCLoseCentroid(dataSet[i,:-1],centroids)
#对比一行到每个中心点的距离，返回距离最短的中心点的label
def getLabelFromCLoseCentroid(dataSetRow,centroids):
    #初始化label为中心点第一点的label
    label = centroids[0,-1]
    #初始化最小值为当前行到中心点第一点的距离值
    #np.linalg.norm计算两个向量的距离
    minDist  = np.linalg.norm(dataSetRow - centroids[0,:-1])
    #对中心点的每个点开始循环
    for i in range(1,centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i,:-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i,-1]
    print "minDist",minDist
    return label
# 更新中心点
#参数：数据集（包含标签），k个分类

def getCentroids(dataSet,k):
    #初始化新的中心点矩阵
    result = np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        #找出最后一列类别为i 的行集,即求一个类别里面的所有点
        oneCluster = dataSet[dataSet[:,-1]==i,:-1]
        #axis = 0 对行求均值，并赋值
        result[i-1,:-1] = np.mean(oneCluster,axis=0)
        result[i-1,-1] = i
    return result
x1 = np.array([1,1])
x2 = np.array([2,1])
x3 = np.array([4,3])
x4 = np.array([5,4])
testX = np.vstack((x1,x2,x3,x4))#将点排列成矩阵
print testX
result =  kmeans(testX,2,10)
print "final result:"
print result

