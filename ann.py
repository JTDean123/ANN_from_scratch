# --------
#
# Construction of an ANN from scratch
# Jason Dean
# April 29th, 2017
# jasontdean.com
#
# --------

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import random
random.seed(12345)
import pandas as pd


# -------- ANN class and functions --------
class Network(object):
    # this class contains the ann

    def __init__(self, inputs, hidden, out):
        # inputs, hidden, and out are the number of nodes in each layer
        self.inputs = inputs
        self.hidden = hidden
        self.out = out
        # initialize parameters to random numbers
        self.W1 = np.random.rand(inputs, hidden)
        self.W2 = np.random.rand(hidden, out)
        self.B1 = np.random.rand(1, hidden)
        self.B2 = np.random.rand(1, out)

    def model(self, inputs, labels, rate):

        # create list to hold total loss at each iteration
        losses = []

        # iterate forward and backwards 100 times to learn parameters
        for i in range(0, 100):
            # compute output from input layer
            Z1 = np.dot(inputs, self.W1) + self.B1
            # pass through activation function (ReLU) of the hidden layer
            A1 = activation(Z1)

            # compute output from hidden layer
            self.Z2 = np.dot(A1, self.W2) + self.B2
            # perform softmax activation of output layer
            A2 = softmax(self.Z2)

            # determine the loss and output to the terminal
            loss_i = loss(A2, labels)
            losses.append(loss_i)

            # ---- back propogation ----
            # determine gradient for descent
            deltaLoss = scoreGradient(A2, labels)

            # back propogate from output
            deltaW2 = np.dot(A1.T, deltaLoss)
            deltaB2 = np.sum(deltaLoss, axis=0, keepdims=True)

            # back propogate to hidden layer
            delta2 = np.dot(deltaLoss, self.W2.T)
            delta2[A1 <= 0] = 0

            # back propogate to input layer
            deltaW1 = np.dot(inputs.T, delta2)
            deltaB1 = np.sum(delta2, axis=0)

            # adjust the parameters based on gradient descent
            self.W1 -= rate * deltaW1
            self.B1 -= rate * deltaB1

            self.W2 -= rate * deltaW2
            self.B2 -= rate * deltaB2

        return losses

    def predict(self, labels):
        # generate predictions based on the highest predicted class probability
        predictions = softmax(self.Z2)

        output = []
        for i in range(len(predictions)):
            if predictions[i].argmax() == labels[i]:
                output.append(1)
            else:
                output.append(0)

        return output

    def boundary(self, meshX):
        # predict a class based on feature data
        Z1 = np.dot(meshX, self.W1) + self.B1
        # pass through activation function (ReLU) of the hidden layer
        A1 = activation(Z1)

        Z2 = np.dot(A1, self.W2) + self.B2
        # perform softmax activation of output layer
        A2 = softmax(Z2)

        return np.argmax(A2, axis=1)


def softmax(score_output):
    # determine a normalized class probability via SoftMax
    scoresExp = np.exp(score_output)
    scoresNorm = scoresExp / np.sum(scoresExp, axis=1, keepdims=True)
    return scoresNorm

def activation(z):
    # ReLu activation function
    z = np.maximum(0, z)
    return z

def loss(probs, labels):
    # determine the cross entropy loss
    totalLoss = 0
    for i, j in zip(probs, labels):
        totalLoss += -np.log(i[j])
    return totalLoss

def scoreGradient(probs, y):
    # determine the error gradient for the output
    for i in range(0, len(probs)):
        probs[i, y[i]] -= 1
    probs = probs / len(probs)
    return probs


# -------- main --------
def run():

    # generate 1000 random observations belonging to one of three classes
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()

    # determine the total error at each learning iteration for 5 different hidden layer node sizes
    errors = pd.DataFrame()

    hidden_nodes = [2, 5, 10, 20, 100]
    for i in hidden_nodes:
        ann = Network(2, i, 3)
        errors_ann = ann.model(X, y, .5)
        errors[str(i)] = errors_ann

    # plot the errors
    font = {'weight': 'bold', 'size': 16}
    plt.rc('font', **font)
    errors.plot()
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Error vs. Number of Hidden Nodes')
    plt.figure(figsize=(10, 10))
    plt.show()
    plt.rcdefaults()

    # make a prediction based on calculated probabilities and plot
    prediction = ann.predict(y)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=prediction, cmap='bwr')
    plt.show()

    # calculate the accuracy of the predictions
    sum(prediction)/len(prediction)

    # generate predictions across a mesh to visualize boundaries
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    boundaries = ann.boundary(mesh)
    boundaries = boundaries.reshape(xx.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, boundaries, cmap='Pastel2')
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.rcdefaults()

# -------- go time --------
if __name__ == '__main__':
    run()