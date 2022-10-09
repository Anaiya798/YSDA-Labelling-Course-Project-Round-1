import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import cfg, get_cosine_power_annealing_scheduler
from backbone.micronet import MicroNet, SwishLinear
from collections import OrderedDict

import os.path
from typing import Tuple, Callable, Optional

import numpy as np
import torchvision.transforms.functional

import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import trange


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

    def __getitem__(self, index) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        image_path, label = self.data.iloc[index]

        image = cv2.imread(os.path.join(self.base_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, torch.FloatTensor([label])

    def calculate_mean_and_std(self):
        """
        Calculates mean and std of the dataset
        """
        dataset_mean = torch.zeros(3)
        dataset_std = torch.zeros(3)
        nb_samples = 0

        for _ in range(5):
            for image, _ in self:
                if isinstance(image, np.ndarray):
                    image = torchvision.transforms.functional.to_tensor(image)

                nb_samples += 1
                dataset_mean += image.mean(dim=(1, 2))
                dataset_std += image.std(dim=(1, 2))

        dataset_mean /= nb_samples
        dataset_std /= nb_samples
        return dataset_mean, dataset_std


def val(dataloader: DataLoader, model_: torch.nn.Module, loss_fn_: Callable, device_: torch.device):
    """Test the `model` on the given dataset."""
    model_.eval()

    n_batches = len(dataloader)
    size = len(dataloader.dataset)
    loss = 0
    tp_, fp_, tn_, fn_ = 0, 0, 0, 0

    all_outputs_ = []
    all_labels_ = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device_), y.to(device_)
            output = model_(X)
            loss += loss_fn_(output, y).item()
            tp_ += ((output > 0.5) & (y == 1)).sum().item()
            fp_ += ((output > 0.5) & (y == 0)).sum().item()
            tn_ += ((output <= 0.5) & (y == 0)).sum().item()
            fn_ += ((output <= 0.5) & (y == 1)).sum().item()
            all_outputs_.append(output.cpu())
            all_labels_.append(y.cpu())

    loss /= n_batches
    tp_ /= size
    fp_ /= size
    tn_ /= size
    fn_ /= size

    all_outputs_ = torch.cat(all_outputs_, dim=0)
    all_labels_ = torch.cat(all_labels_, dim=0)

    return loss, tp_, fp_, tn_, fn_, all_outputs_, all_labels_


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Calculate mean and std of the dataset
    transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224),
        A.ShiftScaleRotate(p=0.5, rotate_limit=5),
        A.ColorJitter(hue=0.02),
        # next augmentation didn't work for my albumentation version
        # A.PixelDropout(drop_value=None, dropout_prob=0.02),
    ])
    # right path to train there
    train_dataset = CarNumbersDataset('..', transform=transform)
    print('Calculating mean and std of the dataset...')
    mean, std = train_dataset.calculate_mean_and_std()
    print(f'Mean: {mean}, std: {std}')

    # Calculate dataset class balance
    print('Calculating dataset class balance...')
    neg, pos = 0, 0
    for _, label in train_dataset:
        if label == 0:
            neg += 1
        else:
            pos += 1
    ratio = neg / pos
    print(f'Dataset class balance: positive: {pos}, negative: {neg}, ratio: {ratio}')

    # Make test dataset
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    print('Making test dataset...')
    # right path to validation there
    test_dataset = CarNumbersDataset('..', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model
    print('Creating model...')
    model = MicroNet(cfg)
    model.classifier[2] = SwishLinear(inp=1024, oup=1)
    # right path to checkpoint, used micronet-m3_90_jcfsc44e.pt
    module_state = torch.load('./checkpoints/..')
    model.load_state_dict(module_state)
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=ratio * torch.ones(1).to(device))

    _, _, _, _, _, all_outputs, all_labels = val(test_loader, model, loss_fn, device)

    t_np = all_outputs.numpy()  # convert to Numpy array
    df = pd.DataFrame(t_np)  # convert to a dataframe
    df.to_csv('val_contest.csv', index=False)  # save to file
