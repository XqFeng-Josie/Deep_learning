# coding=utf-8
# 来自：深度学习进阶：算法与应用（麦子学院）
# 使用传统机器学习进行识别，使用ＳＶＭ分类器

import mnist_loader
# Third-party libraries
# 导入分类器支持向量机SVM
from sklearn import svm
def svm_baseline():
    # 加载数据
    training_data, validation_data, test_data = mnist_loader.load_data()
    # 开始训练
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # 根据测试集预测
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))


if __name__ == "__main__":
    svm_baseline()

