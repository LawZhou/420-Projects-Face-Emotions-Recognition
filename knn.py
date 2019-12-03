# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
from joblib import dump, load


def readNumpys():
    imgs_data = np.load('./numpy_data/classify_data.npy')
    labels_data = np.load('./numpy_data/classify_labels.npy')
    imgs_data = imgs_data.reshape(imgs_data.shape[0],-1)
    return imgs_data, labels_data

def shuffleData(imgs_data, labels_data):
    indices_tmp = np.arange(imgs_data.shape[0], step=3)
    np.random.shuffle(indices_tmp)
    indices = []
    for index in indices_tmp:
        indices.extend([index, index+1, index+2])
    indices = np.array(indices)
    imgs_data = imgs_data[indices]
    labels_data = labels_data[indices]
    return imgs_data, labels_data

def splitTraingTest(imgs_data, labels_data, split_indices):
    val_split = split_indices
    train_imgs_data = imgs_data[val_split:, :]
    val_imgs_data = imgs_data[:val_split, :]

    train_labels_data = labels_data[val_split:]
    val_labels_data = labels_data[:val_split]

    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data

imgs_data , labels_data = readNumpys()

imgs_data , labels_data = shuffleData(imgs_data, labels_data)



trainData, trainLabels, testData, testLabels = splitTraingTest(imgs_data, labels_data, 120)

trainData, trainLabels, valData, valLabels = splitTraingTest(trainData, trainLabels, 120)




# show the sizes of each data split

print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

kVals = range(1, 100)
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
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

model = best_model

dump(model, '/knn_models/knn_model.joblib')
model = load('/knn_models/knn_model.joblib')



predictions = model.predict(testData)

print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

score = model.score(testData, testLabels)
print("The best model achieved highest accuracy of %.2f%% on test data" % (score * 100))


