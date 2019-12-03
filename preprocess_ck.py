import cv2 as cv
import os
import glob
import numpy as np


def makeDataSet(path, label):
    images = []
    labels = []
    path = path + '/*.png'
    for file in glob.glob(path):
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(label)
    return np.array(images, dtype = 'uint8'), np.array(labels, dtype = 'int64')

def createDataSet():
    '''
    Random shuffle data and split training and validation set
    '''

    anger_path = os.path.join(dir, 'anger')
    disgust_path = os.path.join(dir, 'disgust')
    fear_path = os.path.join(dir, 'fear')
    happy_path = os.path.join(dir, 'happy')
    sadness_path = os.path.join(dir, 'sadness')
    surprise_path = os.path.join(dir, 'surprise')
    contempt_path = os.path.join(dir, 'contempt')

    anger_imgs, anger_labels = makeDataSet(anger_path, 0)
    comtempt_imgs, comtempt_labels = makeDataSet(contempt_path, 1)
    disgust_imgs, disgust_labels = makeDataSet(disgust_path, 2)
    fear_imgs, fear_labels = makeDataSet(fear_path, 3)
    happy_imgs, happy_labels = makeDataSet(happy_path, 4)
    sad_imgs, sad_labels = makeDataSet(sadness_path, 5)
    surprise_imgs, surprise_labels = makeDataSet(surprise_path, 6)

    imgs_data = np.vstack((anger_imgs, comtempt_imgs, disgust_imgs,
                           fear_imgs, happy_imgs, sad_imgs, surprise_imgs))
    labels_data = np.hstack((anger_labels, comtempt_labels, disgust_labels, fear_labels, happy_labels, sad_labels,
                             surprise_labels))
    imgs_data, labels_data = shuffleData(imgs_data, labels_data)

    train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = splitTraingTest(imgs_data, labels_data, 60)
    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data

def shuffleData(imgs_data, labels_data):
    '''
    shuffle data in groups of 3
    :param imgs_data:
    :param labels_data:
    :return: shuffled data
    '''
    indices_tmp = np.arange(imgs_data.shape[0], step=3)
    np.random.shuffle(indices_tmp)
    indices = []
    for index in indices_tmp:
        indices.extend([index, index+1, index+2])
    indices = np.array(indices)
    output_imgs_data = imgs_data[indices].copy()
    output_labels_data = labels_data[indices].copy()
    return output_imgs_data, output_labels_data


def splitTraingTest(imgs_data, labels_data, split_indices):
    val_split = split_indices
    train_imgs_data = imgs_data[val_split:, :]
    val_imgs_data = imgs_data[:val_split, :]

    train_labels_data = labels_data[val_split:]
    val_labels_data = labels_data[:val_split]

    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data



def saveNumpys(train_imgs_data, train_labels_data, val_imgs_data, val_labels_data):
    if not os.path.exists('./numpy_data'):
        os.makedirs('./numpy_data')

    np.save('./numpy_data/train_imgs_data.npy', train_imgs_data)
    np.save('./numpy_data/train_labels_data.npy', train_labels_data)
    np.save('./numpy_data/val_imgs_data.npy', val_imgs_data)
    np.save('./numpy_data/val_labels_data.npy', val_labels_data)


def readNumpys():
    train_imgs_data = np.load('./numpy_data/train_imgs_data.npy')
    train_labels_data = np.load('./numpy_data/train_labels_data.npy')
    val_imgs_data = np.load('./numpy_data/val_imgs_data.npy')
    val_labels_data = np.load('./numpy_data/val_labels_data.npy')
    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data

if __name__ == '__main__':

    dir = "CK+"
    train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = createDataSet()
    saveNumpys(train_imgs_data, train_labels_data, val_imgs_data, val_labels_data)

    train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = readNumpys()

    print(train_imgs_data.shape)
    print(train_labels_data.shape)
    print(val_imgs_data.shape)
    print(val_labels_data.shape)

