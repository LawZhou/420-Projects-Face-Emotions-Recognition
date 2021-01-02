import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from ImageDataLoader import imagesDataSet
from torchvision import transforms

from helpers import mainLoop, splitTraingTest, saveModel, shuffleHelper
from model import ResCNN


class EmotionsClassifier:

    def __init__(self, num_epochs=20, train_batch_size=128, test_batch_size=20, learn_rate=0.001, num_class=7):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learn_rate = learn_rate
        self.num_class = num_class

    def train_models(self, data_easy, data_hard):
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set_easy = imagesDataSet(data_easy[0], data_easy[1], transform=trans)
        val_set_easy = imagesDataSet(data_easy[2], data_easy[3], usage="testing", transform=trans)
        trainloader_easy = torch.utils.data.DataLoader(train_set_easy, batch_size=self.train_batch_size, shuffle=True)
        testloader_easy = torch.utils.data.DataLoader(val_set_easy, batch_size=self.test_batch_size, shuffle=True)

        train_set_hard = imagesDataSet(data_hard[0], data_hard[1], transform=trans)
        val_set_hard = imagesDataSet(data_hard[2], data_hard[3], usage="testing", transform=trans)
        trainloader_hard = torch.utils.data.DataLoader(train_set_hard, batch_size=24, shuffle=True)
        testloader_hard = torch.utils.data.DataLoader(val_set_hard, batch_size=4, shuffle=True)

        model_easy = ResCNN(64, 7, 3).to(self.device)
        model_hard = ResCNN(64, 7, 3).to(self.device)

        # Cross entropy loss function:
        criterion_easy = nn.CrossEntropyLoss()
        criterion_hard = nn.CrossEntropyLoss()

        optimizer_easy = torch.optim.Adam(model_easy.parameters(), lr=self.learn_rate)
        scheduler_easy = lr_scheduler.StepLR(optimizer_easy, step_size=150, gamma=0.5)

        optimizer_hard = torch.optim.Adam(model_hard.parameters(), lr=0.0001)
        scheduler_hard = lr_scheduler.StepLR(optimizer_hard, step_size=200, gamma=0.9)

        model_easy, _ = mainLoop(self.learn_rate, self.num_epochs, self.device, model_easy, trainloader_easy,
                              testloader_easy, criterion_easy, scheduler_easy, optimizer_easy)
        model_hard, _ = mainLoop(self.learn_rate, self.num_epochs, self.device, model_hard, trainloader_hard,
                              testloader_hard, criterion_hard, scheduler_hard, optimizer_hard)

        return model_easy, model_hard

    def run(self, model_knn, imgs, labels):
        data_easy, data_hard = divideUpEasyHardData(model_knn, imgs, labels)
        model_easy, model_hard = self.train_models(data_easy, data_hard)
        return model_easy, model_hard

def divideUpEasyHardData(model_knn, imgs, labels):
    knn_input = imgs.reshape(imgs.shape[0], -1)
    difficulties_lables = model_knn.predict(knn_input)
    imgs, labels, difficulties_lables = shuffleData(imgs, labels, difficulties_lables)

    imgs_easy = imgs[np.where(difficulties_lables == 0)[0]]
    labels_easy = labels[np.where(difficulties_lables == 0)[0]]

    imgs_hard = imgs[np.nonzero(difficulties_lables)]
    labels_hard = labels[np.nonzero(difficulties_lables)]

    easy_data = splitTraingTest(imgs_easy, labels_easy, 60)
    hard_data = splitTraingTest(imgs_hard, labels_hard, 48)

    hard_data = (np.vstack((imgs_easy, hard_data[0])), np.hstack((labels_easy, hard_data[1])), hard_data[2], hard_data[3])

    return easy_data, hard_data

def shuffleData(imgs_data, labels_data, difficulties_lables):
    output_imgs_data, output_labels_data, indices = shuffleHelper(imgs_data, labels_data)
    output_diffifulties_labels = difficulties_lables[indices].copy()
    return output_imgs_data, output_labels_data, output_diffifulties_labels
