import numpy as np

class AdelineGD(object):
    """ADAptative LInear NEuron classifier
    Parameters
    --------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    --------------------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data

        Parameters
        ---------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values

        Returns
        ---------------------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            #print('iter:', self.n_iter, ' / output: ', output, ' / errors: ', errors, ' / self.net_input(X): ', self.net_input(X))
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        net_input = np.dot(X, self.w_[1:]) + self.w_[0]
        #net_input = np.dot(X, self.w_[1:])
        return net_input

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, xi):
        """Return class label after unit step"""
        return np.where(self.net_input(xi) >= 0.0, 1, -1)
