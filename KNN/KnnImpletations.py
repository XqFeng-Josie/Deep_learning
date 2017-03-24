#KNN算法实现

import csv
import random
import math
import operator
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #print (dataset)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
def caldistance(instance1,instance2,length):
    distance = 0
   # print (instance1[1],instance1[0])
    for x in range(length):

        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)
def getNeighbors(trainingSet,testInstance,k):
    distance = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = caldistance(testInstance,trainingSet[x],length)
        distance.append((trainingSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors  = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        else:
            pass
    return (correct/float(len(testSet)))*100.0
def main():
    trainingSet = []
    testSet = []
    split = 0.65# 训练集和测试集的数量影响准确率
    loadDataset(r'F:\pycharmprojects\KNN\iris_data.txt',split,trainingSet,testSet)
    print('Trainset = '+repr(trainingSet))
    print('Testset = '+repr(testSet))
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ("predicted="+repr(result)+",action="+repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print ("Accuracy:"+repr(accuracy)+'%')
main()
