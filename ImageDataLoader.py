from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
import numpy as np

######################################################################
# DATA LOADER
######################################################################

class imagesDataSet(Dataset):
    def __init__(self, input_images, output, usage = "training", transform=None):
        # data augmentation in trainging set
        if usage == "training":
            self.input_images, self.output = augmentation(input_images, output)
        elif usage == "testing":
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

def augmentation(train_images, train_labels):
    images = []
    labels = []
    for i in range(len(train_images)):
        img = train_images[i]
        label = train_labels[i]
        # original image
        images.append(img)
        labels.append(label)
        # Horizontal Flip
        horizontal_flip = cv.flip(img, 1)
        images.append(horizontal_flip)
        labels.append(label)
        # rotated image
        rotation = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        images.append(rotation)
        labels.append(label)
        # flip image horizontally and vertically
        flip = cv.flip(img, -1)
        images.append(flip)
        labels.append(label)
        # increase contrast of the image
        contrast = cv.equalizeHist(img)
        images.append(contrast)
        labels.append(label)
    return np.array(images, dtype = 'uint8'), np.array(labels, dtype = 'int64')