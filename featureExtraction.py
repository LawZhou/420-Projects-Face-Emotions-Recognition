import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from countDataLoader import countWrongPredictionsDataSet
from model import ResCNN
from kfoldDataLoaderCK import KFoldImageLoaderCK
from torch.utils.data import Dataset
from helpers import shuffleData, runValidation, mainLoop


class TrainKFoldModels:
    def __init__(self, num_epochs=20, train_batch_size=128, test_batch_size=20, learn_rate=0.001, num_class=7,
                 k_fold=7, M=5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learn_rate = learn_rate
        self.num_class = num_class
        self.k_fold = k_fold
        self.M = M


    ######################################################################
    # MAIN LOOP
    ######################################################################

    def train_model(self, train_imgs, train_labels):
        '''
        After shuffling data, run one k-fold on the data to produce k classifiers
        :return: k classifiers stored in a list
        '''
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        models_store = []
        # shuffe the image and labels for the current k fold iteration
        train_pixels_shf, train_labels_shf = shuffleData(train_imgs, train_labels)

        for i in range(self.k_fold):
            print("%dth fold" % (i + 1))
            train_set = KFoldImageLoaderCK(train_pixels_shf, train_labels_shf, i, 'training', trans)
            val_set = KFoldImageLoaderCK(train_pixels_shf, train_labels_shf, i, 'testing', trans)
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)  # ck+
            testloader = torch.utils.data.DataLoader(val_set, batch_size=self.test_batch_size, shuffle=True)
            model = ResCNN(64, 7, 3).to(self.device)
            # Cross entropy loss function:
            criterion = nn.CrossEntropyLoss()
            model, test_loss = mainLoop(self.learn_rate, self.num_epochs, self.device, model, trainloader, testloader,
                                        criterion)
            models_store.append(model)
        return models_store

    def train_K_Fold_models(self, train_imgs, train_labels):
        '''
        Run M times K-fold to produce M*k models
        :param M: The number of K fold iterations.
        :return: K * M different models for distinguishing easy / hard to classify images
        '''

        all_K_fold_models = []

        for i in range(self.M):
            print("======================Start training %dth K-fold====================" % (i + 1))
            models_store = self.train_model(train_imgs, train_labels)
            all_K_fold_models.extend(models_store)

        return all_K_fold_models


######################################################################
# Run Model
######################################################################

class CountWrongPredictions:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def count_wrong_prediction_times(self, train_imgs, train_labels, all_K_fold_models):
        '''
        We run validation on every model produced by running M times of K-fold, so there are M*k models in total.
        Count numbers of incorrect prediction made by models for every image
        :return: numbers of incorrect prediction made by models for every image
        '''
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        data_set = countWrongPredictionsDataSet(train_imgs, train_labels, trans)
        dataloader = torch.utils.data.DataLoader(data_set, batch_size=len(train_imgs))
        incorrect_countings = np.zeros((train_imgs.shape[0]), dtype=int)
        for _, labels in dataloader:
            # get all labels
            labels = labels.to(self.device)
            for model in all_K_fold_models:
                criterion = nn.CrossEntropyLoss()
                _, _, outputs = runValidation(criterion, self.device, dataloader, model)
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                # Evaluate the result of the prediction
                # Correct prediction = 1, incorrect prediction = 0
                equals = equals.int()
                equals = tensorToNumpy(self.device, equals).reshape(equals.shape[0])
                incorrects = np.where(equals == 0)[0]
                incorrect_countings = self.modifyCountingArray(incorrect_countings, incorrects)
        labels = self.makeClassifyData(incorrect_countings)
        return train_imgs, labels

    def modifyCountingArray(self, countings, incorrects):
        countings[incorrects] += 1
        return countings

    def makeClassifyData(self, countings):
        labels = np.zeros((countings.shape[0]), dtype=int)
        labels[np.nonzero(countings)] = 1
        # saveNumpys(train_imgs, labels)
        return labels

######################################################################
# Helper function:
######################################################################
def tensorToNumpy(device, tensor):
    if device == torch.device("cuda:0"):
        out = tensor.cpu()
    else:
        out = tensor
    result = np.array(out.data)
    return result


