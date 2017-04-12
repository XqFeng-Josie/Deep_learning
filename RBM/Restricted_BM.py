#coding=utf-8
#贝努力受限玻尔兹曼机算法实现
#RBM 学习特征，Logistic_Regression实现分类
#Author：Fengxiaoqin
#License:BSD
print (__doc__)
import numpy as np
import matplotlib.pyplot as plt#图形化方程

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics#模型，数据集，衡量标准
from sklearn.cross_validation import train_test_split#拆分数据集
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

#扩展数据集
def nudge_dataset(X, Y):
    direction_vectors = [
    [[0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 1],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 0],
    [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',weights=w).ravel()
    X = np.concatenate([X] +
    [np.apply_along_axis(shift, 1, X, vector)
    for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y
#加载数据
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=0.2,
random_state=0)
#声明各类模型
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

#开始训练
#设置参数。
#参数应该是由交叉验证,GridSearchCV选出来，但是程序节约时间没有使用交叉验证
rbm.learning_rate = 0.06
rbm.n_iter = 20
#更多的components会取得更好的预测效果，但是也会占用更多的训练时间
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train,Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train,Y_train)


print()
#打印利用RBM学习出来的特征进行判别的结果
print("Logistic regression using RBM features:\n%s\n" % (
metrics.classification_report(
Y_test,
classifier.predict(X_test))))
#打印利用图像自己的像素特征学习出来再进行判别的结果
print("Logistic regression using raw pixel features:\n%s\n" % (
metrics.classification_report(
Y_test,
logistic_classifier.predict(X_test))))

plt.figure(figsize=(4.2, 4))#设置显示页面尺寸大小
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)#定义显示矩阵分布
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
    interpolation='nearest')#设置显示图像属性
    plt.xticks(())
    plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)#设置标题
    #plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)#设置显示页面内的边框距离

plt.show()