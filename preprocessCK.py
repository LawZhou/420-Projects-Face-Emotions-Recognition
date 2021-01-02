import cv2 as cv
import os
import glob
import numpy as np

from helpers import shuffleData, splitTraingTest


class PreprocessCK:
    def __init__(self, dir):
        self.dir = dir


    @staticmethod
    def makeDataSet(path, label):
        images = []
        labels = []
        path = path + '/*.png'
        for file in glob.glob(path):
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label)
        return np.array(images, dtype='uint8'), np.array(labels, dtype='int64')


    def createDataSet(self):
        '''
        Random shuffle data and split training and validation set
        '''

        anger_path = os.path.join(self.dir, 'anger')
        disgust_path = os.path.join(self.dir, 'disgust')
        fear_path = os.path.join(self.dir, 'fear')
        happy_path = os.path.join(self.dir, 'happy')
        sadness_path = os.path.join(self.dir, 'sadness')
        surprise_path = os.path.join(self.dir, 'surprise')
        contempt_path = os.path.join(self.dir, 'contempt')

        anger_imgs, anger_labels = self.makeDataSet(anger_path, 0)
        comtempt_imgs, comtempt_labels = self.makeDataSet(contempt_path, 1)
        disgust_imgs, disgust_labels = self.makeDataSet(disgust_path, 2)
        fear_imgs, fear_labels = self.makeDataSet(fear_path, 3)
        happy_imgs, happy_labels = self.makeDataSet(happy_path, 4)
        sad_imgs, sad_labels = self.makeDataSet(sadness_path, 5)
        surprise_imgs, surprise_labels = self.makeDataSet(surprise_path, 6)

        imgs_data = np.vstack((anger_imgs, comtempt_imgs, disgust_imgs,
                               fear_imgs, happy_imgs, sad_imgs, surprise_imgs))
        labels_data = np.hstack((anger_labels, comtempt_labels, disgust_labels, fear_labels, happy_labels, sad_labels,
                                 surprise_labels))
        imgs_data, labels_data = shuffleData(imgs_data, labels_data)

        train_imgs_data, train_labels_data, val_imgs_data, val_labels_data = splitTraingTest(imgs_data,
                                                                                             labels_data, 60)
        return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data

    def run_preprocess(self):
        train_imgs, train_labels, val_imgs, val_labels = self.createDataSet()

        print("train images:", train_imgs.shape)
        print("train labels:", train_labels.shape)
        print("validation images:", val_imgs.shape)
        print("validation labels:", val_labels.shape)

        return train_imgs, train_labels, val_imgs, val_labels
