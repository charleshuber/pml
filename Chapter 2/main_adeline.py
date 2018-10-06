#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AdelineGD import AdelineGD
from lib_adeline import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4]
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdelineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-square-error)')
ax[0].set_title('Adeline - Learning rate 0.01')

ada2 = AdelineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-square-error')
ax[1].set_title('Adeline - Learning rate 0.0001')

plt.show()
'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdelineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada, plt=ax[0])

ax[0].set_title('Adeline - Gradient Descent')
ax[0].set_xlabel('sepal length [standardized]')
ax[0].set_ylabel('petal length [standardized]')
ax[0].legend(loc='upper left')

ax[1].plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-square-error')
ax[1].set_title('Adeline - Learning rate 0.01')

plt.show()
