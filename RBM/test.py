#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


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

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
    weights=w).ravel()
    X = np.concatenate([X] +
    [np.apply_along_axis(shift, 1, X, vector)
    for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=0.2,
random_state=0)






logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)




print("Logistic regression using raw pixel features:\n%s\n" % (
metrics.classification_report(
Y_test,
logistic_classifier.predict(X_test))))

# Plotting












