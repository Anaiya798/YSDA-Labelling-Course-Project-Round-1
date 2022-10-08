import os.path
from typing import Tuple, Callable, Optional

import numpy as np
import torchvision.transforms.functional
from torch.utils.data import Dataset

import cv2
import torch
import pandas as pd


class CarNumbersDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        """
        :param data_path: path to the .csv file with image paths and labels
        """
        self.base_path = os.path.dirname(data_path)
        self.transform = transform
        self.data = pd.read_csv(data_path, header=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.FloatTensor, int]:
        image_path, label = self.data.iloc[index]

        image = cv2.imread(os.path.join(self.base_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, int(label)

    def calculate_mean_and_std(self):
        """
        Calculates mean and std of the dataset
        """
        mean = torch.zeros(3)
        std = torch.zeros(3)
        nb_samples = 0

        for image, label in self:
            if isinstance(image, np.ndarray):
                image = torchvision.transforms.functional.to_tensor(image)

            nb_samples += 1
            mean += image.mean(dim=(1, 2))
            std += image.std(dim=(1, 2))

        mean /= nb_samples
        std /= nb_samples
        return mean, std


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    dataset = CarNumbersDataset('../dataset/classification/train.csv')
