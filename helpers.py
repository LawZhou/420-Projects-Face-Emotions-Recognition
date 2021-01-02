import os
import time

import numpy as np
import torch
from joblib import load, dump
from torch.optim import lr_scheduler

from model import ResCNN


def shuffleData(imgs_data, labels_data):
    '''
    shuffle data in groups of 3 because each person has three shots for one emotion
    :param imgs_data:
    :param labels_data:
    :return: shuffled data
    '''
    output_imgs_data, output_labels_data, _ = shuffleHelper(imgs_data, labels_data)
    return output_imgs_data, output_labels_data


def shuffleHelper(imgs_data, labels_data):
    indices_tmp = np.arange(imgs_data.shape[0], step=3)
    np.random.shuffle(indices_tmp)
    indices = []
    for index in indices_tmp:
        indices.extend([index, index + 1, index + 2])
    indices = np.array(indices)
    output_imgs_data = imgs_data[indices].copy()
    output_labels_data = labels_data[indices].copy()
    return output_imgs_data, output_labels_data, indices


def splitTraingTest(imgs_data, labels_data, split_indices):
    val_split = split_indices
    train_imgs_data = imgs_data[val_split:, :]
    val_imgs_data = imgs_data[:val_split, :]

    train_labels_data = labels_data[val_split:]
    val_labels_data = labels_data[:val_split]

    return train_imgs_data, train_labels_data, val_imgs_data, val_labels_data


def train(device, num_epochs, epoch, optimizer, trainloader, model, criterion, scheduler=None):
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
        accuracy = calculate_accuracy(accuracy, labels, outputs)
        running_loss += loss.item()
    train_acc = accuracy / len(trainloader)
    print('Training: Epoch [%d/%d] Loss: %.4f, Accuracy: %.4f, Time (s): %d' % (
        epoch + 1, num_epochs, running_loss / len(trainloader), train_acc, time.time() - start))
    return model, running_loss / len(trainloader)


def test(device, num_epochs, model, epoch, testloader, criterion):
    model.eval()
    start = time.time()
    val_acc, val_loss, _ = runValidation(criterion, device, testloader, model)
    time_elapsed = time.time() - start
    print('Testing: Epoch [%d/%d], Val Loss: %.4f, Accuracy: %.4f, Time(s): %d' % (
        epoch + 1, num_epochs, val_loss, val_acc, time_elapsed))
    return val_loss


def runValidation(criterion, device, testloader, model):
    accuracy = 0
    running_loss = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        accuracy = calculate_accuracy(accuracy, labels, outputs)
        running_loss += loss.item()
    val_loss = running_loss / len(testloader)
    val_acc = accuracy / len(testloader)

    return val_acc, val_loss, outputs


def calculate_accuracy(accuracy, labels, outputs):
    ps = torch.exp(outputs)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
    return accuracy


def mainLoop(learn_rate, num_epochs, device, model, trainloader, testloader, criterion, scheduler=None, optimizer=None):
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    if not scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)  # ck+
    best_loss = 1e10
    best_model = None
    for epoch in range(num_epochs):
        model, _ = train(device, num_epochs, epoch, optimizer, trainloader, model, criterion,
                         scheduler=scheduler)
        epoch_loss = test(device, num_epochs, model, epoch, testloader, criterion)
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_model = model
    return best_model, best_loss


######################################################################
    # Save models
######################################################################
def saveModel(model, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder + '/' + filename)
    print("model " + filename + " saved!")


def loadModel(model_name, device):
    # load modals
    model = ResCNN(64, 7, 3)
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    return model

def saveNumpys(train_imgs_data, train_labels_data, val_imgs_data, val_labels_data):
    if not os.path.exists('./numpy_data'):
        os.makedirs('./numpy_data')

    np.save('./numpy_data/train_imgs.npy', train_imgs_data)
    np.save('./numpy_data/train_labels.npy', train_labels_data)
    np.save('./numpy_data/val_imgs.npy', val_imgs_data)
    np.save('./numpy_data/val_labels.npy', val_labels_data)

def readNumpys(img_filename, labels_filename):
    return np.load(img_filename), np.load(labels_filename)

def saveKnnModel(model, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    dump(model, folder + '/' + filename)
    print("knn model " + filename + " saved!")

def readKnnModel(filename):
    model = load(filename)
    return model