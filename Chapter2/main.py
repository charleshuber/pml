#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from lib import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4]
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
"""
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',label='setosa')
plt.scatter(X[50:110, 0], X[50:110, 1], color='blue', marker='x',label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
"""
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

"""
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
"""
plot_decision_regions(X, y, classifier=ppn)
