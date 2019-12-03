import os
import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim import lr_scheduler
from torchvision import transforms
import preprocess_ck
from model import ResCNN
from kfoldDataLoader_ck import kfoldImageLoaderCK
from torch.utils.data import Dataset
from PIL import Image

######################################################################
# Helper function:
######################################################################
def tensorToNumpy(tensor):
    if device == torch.device("cuda:0"):
        out = tensor.cpu()
    else:
        out = tensor
    result = np.array(out.data)
    return result

def modifyCountingArray(countings, incorrects):
    countings[incorrects] += 1
    return countings

def makeClassifyData(countings):
    labels = np.zeros((countings.shape[0]), dtype=int)
    labels[np.nonzero(countings)] = 1
    saveNumpys(train_pixels, labels)

def saveNumpys(pixels, lables):
    if not os.path.exists('./numpy_data'):
        os.makedirs('./numpy_data')
    np.save('./numpy_data/classify_data.npy', pixels)
    np.save('./numpy_data/classify_labels.npy', lables)
######################################################################
# Count Data Loader
######################################################################
class countWrongPredictionsDataSet(Dataset):
    def __init__(self, input_images, output, transform=None):
        self.input_images = input_images
        self.output = output
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]

        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return [image, output]
######################################################################
# Preprocess
######################################################################
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
        accuracy = calculate_accuracy(accuracy, labels, outputs)
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


######################################################################
# MAIN LOOP
######################################################################

def mainLoop(model, num_epochs, learn_rate, trainloader, testloader, device, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    best_loss = 1e10
    scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5) #ck+
    best_model = None
    for epoch in range(num_epochs):
        model, _ = train(epoch, num_epochs, optimizer, trainloader, model, device, criterion,
                                  scheduler=scheduler)
        epoch_loss = test(model, epoch, num_epochs, testloader, device, criterion)
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_model = model
    return best_model, best_loss


def train_model():
    '''
    After shuffling data, run one k-fold on the data to produce k classifiers
    :return: k classifiers stored in a list
    '''
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    models_store = []
    # shuffe the image and labels for the current k fold iteration
    train_pixels_shf , train_labels_shf = shuffleData(train_pixels, train_labels)

    for i in range(k_fold):
        print("%dth fold" % (i+1))
        train_set = kfoldImageLoaderCK(train_pixels_shf , train_labels_shf, i, 'training', trans)
        val_set = kfoldImageLoaderCK(train_pixels_shf , train_labels_shf, i, 'testing', trans)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True) #ck+
        testloader = torch.utils.data.DataLoader(val_set, batch_size=test_batch_size, shuffle=True)
        model = ResCNN(64, 7, 3).to(device)
        # Cross entropy loss function:
        criterion = nn.CrossEntropyLoss()
        model, test_loss = mainLoop(model, num_epochs, learn_rate, trainloader, testloader, device, criterion)
        models_store.append(model)
    return models_store


def train_K_Fold_models(M):
    '''
    Run M times K-fold to produce M*k models
    :param M: The number of K fold iterations.
    :return: K * M different models for distinguishing easy / hard to classify images
    '''

    all_K_fold_models = []

    for i in range(M):
        print("======================Start training %dth K-fold====================" % (i+1))
        models_store = train_model()
        all_K_fold_models.extend(models_store)

    return all_K_fold_models

def saveModels():
    folder = './k_fold_models'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(len(models_store)):
        torch.save(models_store[i].state_dict(), folder + '/' + "k_fold_model_" + str(i))

def loadModels(M):
    models = []
    for i in range(k_fold * M):
        model = ResCNN(64, 7, 3)
        model.load_state_dict(torch.load('./k_fold_models/' + 'k_fold_model_' + str(i)))
        model = model.to(device)
        models.append(model)
    return models
######################################################################
# Run Model
######################################################################
def count_wrong_prediction_times():
    '''
    We run validation on every model produced by running M times of K-fold, so there are M*k models in total.
    Count numbers of incorrect prediction made by models for every image
    :return: numbers of incorrect prediction made by models for every image
    '''
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_set = countWrongPredictionsDataSet(train_pixels, train_labels, trans)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=len(train_pixels))
    countings = np.zeros((train_pixels.shape[0]), dtype=int)
    for _, labels in dataloader:
        # get all labels
        labels = labels.to(device)
        for model in models_store:
            criterion = nn.CrossEntropyLoss()
            _, _, outputs = runValidation(criterion, device, dataloader, model)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # Evaluate the result of the prediction
            # Correct prediction = 1, incorrect prediction = 0
            equals = equals.int()
            equals = tensorToNumpy(equals).reshape(equals.shape[0])
            incorrects = np.where(equals == 0)[0]
            countings = modifyCountingArray(countings, incorrects)
    return countings







if __name__ == '__main__':

    train_pixels, train_labels, val_pixels, val_labels = preprocess_ck.readNumpys()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_epochs = 20
    train_batch_size = 128
    test_batch_size=20
    learn_rate = 0.001
    num_class = 7
    k_fold = 7
    M = 5

    # models_store = train_K_Fold_models(M)
    # saveModels()
    models_store = loadModels(M)
    incorrect_counctings = count_wrong_prediction_times()
    makeClassifyData(incorrect_counctings)
    print(len(models_store))