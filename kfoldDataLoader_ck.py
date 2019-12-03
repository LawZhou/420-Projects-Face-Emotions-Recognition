import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class kfoldImageLoaderCK(Dataset):
    '''
    k fold data set generator
    120 testing
    rest training
    '''
    def __init__(self, input_images, output, numth_fold, split = 'training', transform=None):
        # assume data already pre-shuffle
        group_size = 120
        split_i = numth_fold * group_size
        # Do augmentation on training only
        if split == 'training':
            train_images = np.vstack((input_images[:split_i], input_images[split_i+group_size:]))
            train_labels = np.hstack((output[:split_i], output[split_i+group_size:]))
            self.input_images, self.output = augmentation(train_images, train_labels)
        elif split == 'testing':
            self.input_images = input_images[split_i:split_i+group_size]
            self.output = output[split_i:split_i+group_size]
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