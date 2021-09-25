import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn

count =0

def get_data(noOfSamples=None, number=0):
    if number == 0:
        df = pd.read_csv('mnist_train.csv')
    else:
        df = pd.read_csv('mnist_test.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if noOfSamples is not None:
        X, Y = X[:noOfSamples], Y[:noOfSamples]
    return X, Y

class NaiveBayes(object):
    def fit(self, X, Y, smoothing=10e-4):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var,allow_singular=True) + np.log(self.priors[c])

        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, Y = get_data(60000, 0)
    X1, Y1 = get_data(10000, 1)
    Ntrain = len(Y)
    Ntest = len(Y1)
    Xtrain, Ytrain = X[:int(Ntrain)], Y[:int(Ntrain)]
    Xtest, Ytest = X1[:int(Ntest)], Y1[:int(Ntest)]

    model = NaiveBayes()
    model.fit(Xtrain, Ytrain)

    P = model.predict(Xtest)
    for digit in range(0, 9):
        count = 0
        for i, a in enumerate(P):
            if a != digit and Ytest[i] == digit:
                count += 1
        print(digit, ' - Digit Accuracy (mean) - ', 1-(count / np.count_nonzero(Ytest == digit)))

    print("Total Test Accuracy (mean):", np.mean(P==Ytest))
