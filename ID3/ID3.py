#决策树算法的实现

#DecisionTree只能识别数值型 因此需要进行转换
from sklearn.feature_extraction import DictVectorizer#原始数据转化为数值型
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

filename = "F:\pycharmprojects\ID3\data.csv"
allElectionicsData = open(filename,'rb')
reader = csv.reader(allElectionicsData)
headers = next(reader)
print(headers)
featureList=[]
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])
    for i in range len(row)-1
