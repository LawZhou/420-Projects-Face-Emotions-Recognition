import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ImageDataLoader import imagesDataSet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from helpers import calculate_accuracy, readNumpys, readKnnModel, loadModel


class Evaluator:

    def __init__(self, class_names):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

    @staticmethod
    def divideDataEasyHard(knn_model, pixels, labels):
        '''
        Divide data to easy and hard using our Knn classifier
        :param knn_model:
        :param pixels:
        :param labels:
        :return:Divided data
        '''
        knn_input = pixels.reshape(pixels.shape[0], -1)
        predictions = knn_model.predict(knn_input)
        imgs_easy = pixels[np.where(predictions == 0)[0]]
        labels_easy = labels[np.where(predictions == 0)[0]]

        imgs_hard = pixels[np.nonzero(predictions)]
        labels_hard = labels[np.nonzero(predictions)]

        return (imgs_easy, labels_easy), (imgs_hard, labels_hard)

    def runTest(self, data_easy, data_hard, model_easy, model_hard):
        '''
        Run evaluation on validation data, and plot the confusion matrix
        :return:
        '''
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_set_easy = imagesDataSet(data_easy[0], data_easy[1], usage='testing', transform=trans)
        testloader_easy = torch.utils.data.DataLoader(test_set_easy, batch_size=len(test_set_easy), shuffle=True)
        # Cross entropy loss function:
        criterion = nn.CrossEntropyLoss()
        outputs_easy, labels_easy = self.runValidation(testloader_easy, model_easy)

        test_set_hard = imagesDataSet(data_hard[0], data_hard[1], usage='testing', transform=trans)
        testloader_hard = torch.utils.data.DataLoader(test_set_hard, batch_size=len(test_set_hard), shuffle=True)
        outputs_hard, labels_hard = self.runValidation(testloader_hard, model_hard)

        outputs = torch.cat((outputs_easy, outputs_hard), dim=0)
        labels = torch.cat((labels_easy, labels_hard), dim=0)

        accuracy = calculate_accuracy(0, labels, outputs).item()
        loss = criterion(outputs, labels).cpu().item()

        cm = self.calculateConfusionMatrix(outputs, labels)
        self.plot_confusion_matrix(cm, self.class_names, accuracy)
        print("accuracy: %2f%%" % (accuracy * 100))
        print("loss: ", loss)

    @staticmethod
    def calculateConfusionMatrix(outputs, labels):
        ps = torch.exp(outputs)
        _, predictions = ps.topk(1, dim=1)
        return confusion_matrix(labels.data.cpu().detach().numpy(), predictions.cpu().detach().numpy())

    @staticmethod
    def plot_confusion_matrix(cm, classes, acc,
                              cmap=plt.cm.Reds):
        """
        This function prints and plots the confusion matrix.
        """

        plt.figure(figsize=(10, 8))
        title = 'Confusion Matrix (Accuracy: %0.3f%%)' % (acc * 100)
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if i == 0:
                plt.text(j, i + 0.2, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="black")
            elif i == cm.shape[1] - 1:
                plt.text(j, i - 0.2, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white")
            else:
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="black")

        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.tight_layout()
        plt.show()

    def runValidation(self, testloader, model):
        for images, labels in testloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
        return outputs, labels

    def runEvaluation(self, model_easy, model_hard, knn_model, imgs_filename, labels_filename):
        # read knn model
        model_knn = readKnnModel(knn_model)
        # model_knn = readKnnModel("./knn_models/knn_model_final.joblib")
        # read test data
        imgs_data, lables_data = readNumpys(imgs_filename, labels_filename)

        data_easy, data_hard = self.divideDataEasyHard(model_knn, imgs_data, lables_data)
        # load models
        model_easy = loadModel(model_easy, self.device)
        model_hard = loadModel(model_hard, self.device)

        self.runTest(data_easy, data_hard, model_easy, model_hard)

