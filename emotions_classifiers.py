import os
import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim import lr_scheduler
from ImageDataLoader import imagesDataSet
from torchvision import transforms
from model import ResCNN
from joblib import load



######################################################################
# Helper functions
######################################################################
def readKnnModel(filename):
    model = load(filename)
    return model

def splitTraingVal(imgs_data, labels_data, val_split=24):
    train_imgs_data = imgs_data[val_split:, :]
    val_imgs_data = imgs_data[:val_split, :]

    train_labels_data = labels_data[val_split:]
    val_labels_data = labels_data[:val_split]

    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data

def readNumpys():
    train_imgs_data = np.load('./numpy_data/train_imgs_data.npy')
    train_labels_data = np.load('./numpy_data/train_labels_data.npy')

    return train_imgs_data, train_labels_data

def shuffleData(imgs_data, labels_data, difficulties_lables):
    indices_tmp = np.arange(imgs_data.shape[0], step=3)
    np.random.shuffle(indices_tmp)
    indices = []
    for index in indices_tmp:
        indices.extend([index, index+1, index+2])
    indices = np.array(indices)
    output_imgs_data = imgs_data[indices].copy()
    output_labels_data = labels_data[indices].copy()
    output_diffifulties_labels = difficulties_lables[indices].copy()
    return output_imgs_data, output_labels_data, output_diffifulties_labels


######################################################################
# Preprocess
######################################################################
def divideUpEasyHardData():
    imgs, labels = readNumpys()
    knn_input = imgs.reshape(imgs.shape[0], -1)
    difficulties_lables = model_knn.predict(knn_input)
    imgs, labels, difficulties_lables = shuffleData(imgs, labels,difficulties_lables)

    imgs_easy = imgs[np.where(difficulties_lables == 0)[0]]
    labels_easy = labels[np.where(difficulties_lables == 0)[0]]

    imgs_hard = imgs[np.nonzero(difficulties_lables)]
    labels_hard = labels[np.nonzero(difficulties_lables)]

    easy_data = splitTraingVal(imgs_easy, labels_easy, 60)
    hard_data = splitTraingVal(imgs_hard, labels_hard, 48)

    hard_data = (np.vstack((imgs_easy, hard_data[0])), np.hstack((labels_easy, hard_data[1])), hard_data[2], hard_data[3])

    return easy_data, hard_data







######################################################################
# Train
######################################################################
def train(epoch, num_epochs, optimizer, trainloader, model, device, criterion, scheduler=None):
    start = time.time()
    model.train()  # Change model to 'train' mode
    running_loss = 0
    accuracy = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (scheduler):
            scheduler.step()
        accuracy = calculateAccuracy(accuracy, labels, outputs)
        running_loss += loss.item()
    train_acc = accuracy / len(trainloader)
    print('Training: Epoch [%d/%d] Loss: %.4f, Accuracy: %.4f, Time (s): %d' % (
        epoch + 1, num_epochs, running_loss / len(trainloader), train_acc, time.time() - start))
    return model, running_loss / len(trainloader)


######################################################################
# Test
######################################################################
def test(model, epoch, num_epochs, testloader, device, criterion):
    model.eval()
    start = time.time()
    val_acc, val_loss = runValidation(criterion, device, testloader, model)
    time_elapsed = time.time() - start
    print('Testing: Epoch [%d/%d], Val Loss: %.4f, Accuracy: %.4f, Time(s): %d' % (
        epoch + 1, num_epochs, val_loss, val_acc, time_elapsed))
    return val_loss


def runValidation(criterion, device, testloader, model, store=False):
    accuracy = 0
    running_loss = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        accuracy = calculateAccuracy(accuracy, labels, outputs)
        running_loss += loss.item()
    val_loss = running_loss / len(testloader)
    val_acc = accuracy / len(testloader)
    return val_acc, val_loss


def calculateAccuracy(accuracy, labels, outputs):
    ps = torch.exp(outputs)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
    return accuracy


######################################################################
# MAIN LOOP
######################################################################

def mainLoop(model, num_epochs, trainloader, testloader, device, criterion, scheduler, optimizer):
    best_loss = 1e10
    best_model = None
    for epoch in range(num_epochs):
        model, _ = train(epoch, num_epochs, optimizer, trainloader, model, device, criterion,
                                  scheduler=scheduler)
        epoch_loss = test(model, epoch, num_epochs, testloader, device, criterion)
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_model = model
    return best_model


def train_models():
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set_easy = imagesDataSet(data_easy[0], data_easy[1], transform=trans)
    val_set_easy = imagesDataSet(data_easy[2], data_easy[3], usage="testing", transform=trans)
    trainloader_easy = torch.utils.data.DataLoader(train_set_easy, batch_size=train_batch_size, shuffle=True)
    testloader_easy = torch.utils.data.DataLoader(val_set_easy, batch_size=test_batch_size, shuffle=True)

    train_set_hard = imagesDataSet(data_hard[0], data_hard[1], transform=trans)
    val_set_hard = imagesDataSet(data_hard[2], data_hard[3], usage="testing", transform=trans)
    trainloader_hard = torch.utils.data.DataLoader(train_set_hard, batch_size=24, shuffle=True)
    testloader_hard = torch.utils.data.DataLoader(val_set_hard, batch_size=4, shuffle=True)

    model_easy = ResCNN(64, 7, 3).to(device)
    model_hard = ResCNN(64, 7, 3).to(device)

    # Cross entropy loss function:
    criterion = nn.CrossEntropyLoss()

    optimizer_easy = torch.optim.Adam(model_easy.parameters(), lr=learn_rate)
    scheduler_easy = lr_scheduler.StepLR(optimizer_easy, step_size=150, gamma=0.5)

    optimizer_hard = torch.optim.Adam(model_hard.parameters(), lr=0.0001)
    scheduler_hard = lr_scheduler.StepLR(optimizer_hard, step_size=200, gamma=0.9)


    model_easy = mainLoop(model_easy, num_epochs, trainloader_easy, testloader_easy, device, criterion, scheduler_easy,
                          optimizer_easy)
    model_hard = mainLoop(model_hard, num_epochs, trainloader_hard, testloader_hard, device, criterion, scheduler_hard,
                          optimizer_hard)

    saveModel(model_easy, "emotion_classifier_models", "model_easy.pt")
    saveModel(model_hard, "emotion_classifier_models", "model_hard.pt")


def saveModel(model, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder + '/' + filename)
    print("model " + filename + " saved!")

######################################################################
# Run Model
######################################################################

def loadModel(model_name):
    # load modals
    model = ResCNN(64, 7, 3)
    model.load_state_dict(torch.load('emotion_classifier_models/' + model_name))
    model = model.to(device)
    return model



if __name__ == '__main__':
    # read knn model
    model_knn = readKnnModel("./knn_models/knn_model_final.joblib")

    data_easy, data_hard = divideUpEasyHardData()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_epochs = 2
    train_batch_size = 128
    test_batch_size = 50
    learn_rate = 0.001
    num_class = 7

    train_models()