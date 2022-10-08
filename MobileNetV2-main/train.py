import os.path
from typing import Tuple, Callable, Optional

import numpy as np
import torchvision.transforms.functional
from torchvision.models import MobileNet_V2_Weights

import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import trange

from utils import get_cosine_power_annealing_scheduler

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

        for image, label in self:
            if isinstance(image, np.ndarray):
                image = torchvision.transforms.functional.to_tensor(image)

            nb_samples += 1
            dataset_mean += image.mean(dim=(1, 2))
            dataset_std += image.std(dim=(1, 2))

        dataset_mean /= nb_samples
        dataset_std /= nb_samples
        return dataset_mean, dataset_std


def train(
        dataloader: DataLoader, model_: torch.nn.Module, loss_fn_: Callable, optimizer_: torch.optim.Optimizer,
        scheduler_, device_: torch.device,
):
    """Train the `model` on the given dataset for a single epoch."""
    model_.train()

    n_batches = len(dataloader)
    loss_sum = 0

    for X, y in dataloader:
        def closure():
            optimizer_.zero_grad()
            output = model_(X)
            loss = loss_fn_(output, y)
            loss.backward()
            return loss.item()

        X, y = X.to(device_), y.to(device_)
        loss_sum += optimizer_.step(closure)
        scheduler_.step()

    return loss_sum / n_batches


def test(dataloader: DataLoader, model_: torch.nn.Module, loss_fn_: Callable, device_: torch.device):
    """Test the `model` on the given dataset."""
    model_.eval()

    n_batches = len(dataloader)
    size = len(dataloader.dataset)
    loss = 0
    tp_, fp_, tn_, fn_ = 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device_), y.to(device_)
            output = model_(X)
            loss += loss_fn_(output, y).item()
            tp_ += ((output > 0.5) & (y == 1)).sum().item()
            fp_ += ((output > 0.5) & (y == 0)).sum().item()
            tn_ += ((output <= 0.5) & (y == 0)).sum().item()
            fn_ += ((output <= 0.5) & (y == 1)).sum().item()

    loss /= n_batches
    tp_ /= size
    fp_ /= size
    tn_ /= size
    fn_ /= size

    return loss, tp_, fp_, tn_, fn_


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate mean and std of the dataset
    dataset = CarNumbersDataset('../dataset/classification/train.csv')
    print('Calculating mean and std of the dataset...')
    mean, std = dataset.calculate_mean_and_std()
    print(mean, std)

    # Make train dataset
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    print('Making train dataset...')
    train_dataset = CarNumbersDataset('../dataset/classification/train.csv', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Make test dataset
    test_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    print('Making test dataset...')
    test_dataset = CarNumbersDataset('../dataset/classification/train.csv', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model
    print('Creating model...')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = torch.nn.Linear(1280, 1)
    model = model.to(device)

    # Freeze layers (Optional)
    for param in model.features.parameters():
        param.requires_grad = False

    # Train model and log metrics
    wandb.login()
    with wandb.init(
        entity='ysda-labelling-course-team',
        project='YSDA-Labelling-Course-Project-Round-1',
        config={
            'epochs': 200,
            'start_lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True,
            'batch_size': 32,
            'model': 'MobileNetV2',
            'warmup_steps': 5,
            'num_cycles': 4,
            'lr_gamma': 0.5,
        },
    ):
        config = wandb.config

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.start_lr, momentum=config.momentum,
            weight_decay=config.weight_decay, nesterov=config.nesterov,
        )
        scheduler = get_cosine_power_annealing_scheduler(
            optimizer, config.warmup_steps, config.epochs, config.num_cycles, gamma=config.lr_gamma,
        )

        for epoch in trange(config.epochs):
            train_loss = train(train_loader, model, loss_fn, optimizer, scheduler, device)
            wandb.log({'train_loss': train_loss}, step=epoch)

            if epoch % 10 == 0:
                test_loss, tp, fp, tn, fn = test(train_loader, model, loss_fn, device)
                wandb.log({'test_loss': test_loss, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, step=epoch)

                model_weights_path = f'checkpoints/{config.model}_{epoch}.pt'
                torch.save(model.state_dict(), model_weights_path)
                wandb.save(model_weights_path)
