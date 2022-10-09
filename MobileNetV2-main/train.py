import os.path
from typing import Tuple, Callable, Optional
from sklearn.model_selection import train_test_split

import numpy as np
import torchvision.transforms.functional
from torchvision.models import MobileNet_V2_Weights

import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
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

    @property
    def labels(self) -> np.ndarray:
        return self.data['label'].values


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
    current_lr_ = scheduler.optimizer.param_groups[0]['lr']

    return loss_sum / n_batches, current_lr_


def test(dataloader: DataLoader, model_: torch.nn.Module, loss_fn_: Callable, device_: torch.device):
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
        A.PixelDropout(drop_value=None, dropout_prob=0.02),
    ])
    dataset = CarNumbersDataset('../dataset/classification/train.csv', transform=transform)
    print('Calculating mean and std of the dataset...')
    mean, std = dataset.calculate_mean_and_std()
    print(f'Mean: {mean}, std: {std}')

    # Calculate dataset class balance
    print('Calculating dataset class balance...')
    neg, pos = 0, 0
    for _, label in dataset:
        if label == 0:
            neg += 1
        else:
            pos += 1
    ratio = neg / pos
    print(f'Dataset class balance: positive: {pos}, negative: {neg}, ratio: {ratio}')

    # Split for train and validation
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        shuffle=True,
        stratify=dataset.labels,
    )

    # Make train dataset
    train_transform = A.Compose([
        transform,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    print('Making train dataset...')
    train_dataset = CarNumbersDataset('../dataset/classification/train.csv', transform=train_transform)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=4,
        sampler=train_sampler, pin_memory=True, drop_last=True,
    )

    # Make validation dataset
    val_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    print('Making test dataset...')
    val_dataset = CarNumbersDataset('../dataset/classification/train.csv', transform=val_transform)
    val_sampler = SequentialSampler(val_idx)
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=4,
        sampler=val_sampler, pin_memory=True, drop_last=False,
    )

    # Create model
    print('Creating model...')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = torch.nn.Linear(1280, 1)
    model = model.to(device)

    # Freeze layers (Optional)
    # for param in model.features.parameters():
    #     param.requires_grad = False

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
    ) as run:
        config = wandb.config

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=ratio * torch.ones(1).to(device))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.start_lr, momentum=config.momentum,
            weight_decay=config.weight_decay, nesterov=config.nesterov,
        )
        scheduler = get_cosine_power_annealing_scheduler(
            optimizer, config.warmup_steps, config.epochs, config.num_cycles, gamma=config.lr_gamma,
        )

        for epoch in trange(config.epochs):
            train_loss, current_lr = train(train_loader, model, loss_fn, optimizer, scheduler, device)
            wandb.log({'train_loss': train_loss, 'lr': current_lr}, step=epoch)

            if epoch % 10 == 0:
                test_loss, tp, fp, tn, fn, predictions, ground_truth = test(train_loader, model, loss_fn, device)
                predictions = torch.cat([1 - predictions, predictions], dim=1)

                wandb.log({'test_loss': test_loss, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, step=epoch)
                wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
                wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})

                model_weights_path = f'checkpoints/{config.model}_{epoch}_{run.id}.pt'
                torch.save(model.state_dict(), model_weights_path)
                wandb.save(model_weights_path)
