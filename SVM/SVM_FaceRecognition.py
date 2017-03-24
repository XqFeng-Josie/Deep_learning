#SVM 人脸识别方案实现

#coding=utf-8
from __future__ import print_function
from time import time
import logging #打印程序进展的信息
import matplotlib.pyplot as plt #绘图
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print (__doc__)

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
#下载数据集（人脸数据集）
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
#提取实例数
n_samples, h ,w = lfw_people.images.shape

X = lfw_people.data
#提取特征值数（维度，列数）
n_features = X.shape[1]
#对应不同人的身份
y = lfw_people.target
#类别里的名字
target_names = lfw_people.target_names
#返回类别里的人名数(类别数)
n_classes = target_names.shape[0]

print ("Total data size:")
print ("n_samples: %d"% n_samples)
print ("n_fearures: %d"% n_features)
print ("n_classes: %d"% n_classes)

#拆分数据集为训练集和测试集
#拆分成两个矩阵和两个特征向量，对应测试集或者训练集矩阵以及相对的类别标记
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#PCA降维，将特征值减少
n_components = 150
print ("Extracing the top %d eigenfaces from %d faces" %(n_components,X_train.shape[0]))
t0 = time()#初始时间
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X_train)

print ("done in %0.3fs"%(time()-t0))
#提取人脸的特征
eigenfaces = pca.components_.reshape((n_components,h,w))
print("Projectig the input data on the eigenfaces orthonormal basis")
t0 = time()

#将矩阵转化为更低维的矩阵
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print ("done in %0.3fs"%(time()-t0))

#训练分类器
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],#权重
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],}#特征点使用的比例
#不同参数的组合（C和gamma）建立核函数，进行搜索，找出归类精确度最高的一组（5*6=30中组合）


#建立模型
clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'),param_grid)
#进行训练，找出超平面
clf = clf.fit(X_train_pca,y_train)
print ("done in %0.3fs"%(time()-t0))
print ("Base estimator found by grid search： ")
print (clf.best_estimator_)

#对建好的模型进行测试评估，并建立可视化结果
print ("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print ("done in %0.3fs"%(time()-t0))

print (classification_report(y_test,y_pred,target_names=target_names))
#建立矩阵方格，对角线上的元素越多，准确率越高
print (confusion_matrix(y_test,y_pred,labels=range(n_classes)))

#将预测结果和准确率打印

def plot_gallery(images,titles,h,w,n_row =3,n_col=4):
    plt.figure(figsize=(1.8 * n_col,2.4 * n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=.99,top=.90,hspace=.35)
    for i  in range(n_row * n_col) :
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    return 'predicted : %s\ntrue:     %s'% (pred_name,true_name)
prediction_titles = [title(y_pred,y_test,target_names,i)
                     for i in range(y_pred.shape[0])]
plot_gallery(X_test,prediction_titles,h,w)

eigenfaces_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenfaces_titles,h,w)
plt.show()











