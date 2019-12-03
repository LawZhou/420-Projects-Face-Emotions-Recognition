from joblib import load
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from model import ResCNN
from ImageDataLoader import imagesDataSet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def readKnnModel(filename):
    model = load(filename)
    return model

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

def readNumpys():
    return np.load('./numpy_data/val_imgs_data.npy'), np.load('./numpy_data/val_labels_data.npy')

def loadModel(model_name):
    # load modals
    model = ResCNN(64, 7, 3)
    model.load_state_dict(torch.load('emotion_classifier_models/' + model_name))
    model = model.to(device)
    return model

def runValidation(device, testloader, model):
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
    return outputs, labels

def calculateAccuracy(labels, outputs):
    accuracy = 0
    ps = torch.exp(outputs)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
    return accuracy

def runTest():
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
    outputs_easy, labels_easy = runValidation(device, testloader_easy, model_easy)

    test_set_hard = imagesDataSet(data_hard[0], data_hard[1], usage='testing', transform=trans)
    testloader_hard = torch.utils.data.DataLoader(test_set_hard, batch_size=len(test_set_hard), shuffle=True)
    outputs_hard, labels_hard = runValidation(device, testloader_hard, model_hard)

    outputs = torch.cat((outputs_easy, outputs_hard), dim=0)
    labels = torch.cat((labels_easy, labels_hard), dim=0)

    accuracy = calculateAccuracy(labels, outputs).item()
    loss = criterion(outputs, labels).cpu().item()

    cm = calculateConfusionMatrix(outputs, labels)
    plot_confusion_matrix(cm, class_names, accuracy)
    print("accuracy: %2f%%" % (accuracy * 100))
    print("loss: ", loss)

def calculateConfusionMatrix(outputs, labels):
    ps = torch.exp(outputs)
    _, predictions = ps.topk(1, dim=1)
    return confusion_matrix(labels.data.cpu().detach().numpy(), predictions.cpu().detach().numpy())

def plot_confusion_matrix(cm, classes,acc,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    """

    plt.figure(figsize=(10, 8))
    title = 'Confusion Matrix (Accuracy: %0.3f%%)' %(acc*100)
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
            plt.text(j, i+0.2, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="black")
        elif i == cm.shape[1] - 1:
            plt.text(j, i-0.2, format(cm[i, j], fmt),
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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    #read knn model
    model_knn = readKnnModel("./knn_models/knn_model_final.joblib")
    #read test data
    imgs_data, lables_data = readNumpys()

    data_easy, data_hard = divideDataEasyHard(model_knn, imgs_data, lables_data)
    #load models
    model_easy = loadModel("model_easy.pt")
    model_hard = loadModel("model_hard.pt")



    test_batch_size = 20

    runTest()




