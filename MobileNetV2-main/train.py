import os.path
from typing import Tuple, Callable, Optional
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

        image = torch.from_numpy(image).float()

        return image, int(label)


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    dataset = CarNumbersDataset('../dataset/classification/train.csv')
