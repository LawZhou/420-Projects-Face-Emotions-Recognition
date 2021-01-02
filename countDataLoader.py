import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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