# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
from helpers import shuffleData, splitTraingTest

class KNN:

    def __init__(self, kVals=range(1, 100)):
        self.kVals = kVals
        self.accuracies = []
        self.best_model = None

    def train(self, imgs, labels):
        imgs_data , labels_data = shuffleData(imgs, labels)
        imgs_data = imgs_data.reshape(imgs_data.shape[0], -1)
        trainData, trainLabels, testData, testLabels = splitTraingTest(imgs_data, labels_data, 120)
        trainData, trainLabels, valData, valLabels = splitTraingTest(trainData, trainLabels, 120)
        # show the sizes of each data split
        print("training data points: {}".format(len(trainLabels)))
        print("validation data points: {}".format(len(valLabels)))
        print("testing data points: {}".format(len(testLabels)))
        # initialize the values of k for our k-Nearest Neighbor classifier along with the
        # list of accuracies for each value of k
        accuracies = []
        best_model = None
        # loop over various values of `k` for the k-Nearest Neighbor classifier
        for k in range(1, 100):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(trainData, trainLabels)
            # evaluate the model and update the accuracies list
            score = model.score(valData, valLabels)
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            if len(accuracies) > 0:
                if score > max(accuracies):
                    best_model = model
            accuracies.append(score)
        # find the value of k that has the largest accuracy
        i = np.argmax(accuracies)
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (self.kVals[i],
                                                                               accuracies[i] * 100))
        model = best_model
        predictions = model.predict(testData)
        print("EVALUATION ON TESTING DATA")
        print(classification_report(testLabels, predictions))
        score = model.score(testData, testLabels)
        print("The best model achieved highest accuracy of %.2f%% on test data" % (score * 100))
        return model

