import numpy as np
import pandas as pd
import os
import json
import random
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from albumentations import *
from albumentations.pytorch import ToTensorV2
from model import MultiOutputEffNet
import cv2


class CustomDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        image = np.array(Image.open(os.path.join(root, 'train_dataset', self.image_files[i])).convert('RGB'))
        labels = self.labels[i]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, labels[0], labels[1]


def train_fn(i, model, data_loader, loss_fn, optimizer, device):
    overall_loss = 0.
    with tqdm(data_loader, total=len(data_loader), desc=f'Training, phase {i} :') as loader:
        for data, weather in loader:
            optimizer.zero_grad()
            data, weather = data.to(device), weather.to(device)
            output = model(data)
            loss = loss_fn(output, weather)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
            loader.set_postfix(loss=overall_loss / len(data_loader))


def validation_fn(i, model, data_loader, device):
    model.eval()
    weather_accuracy = 0.
    with tqdm(data_loader, total=len(data_loader), desc=f'Validating, phase {i} :') as loader:
        with torch.no_grad():
            for data, weather in loader:
                data, weather = data.to(device), weather.to(device)

                output = model(data)

                weather_accuracy += accuracy_score(weather.cpu().detach().numpy(),
                                                   torch.argmax(output, dim=1).cpu().detach().numpy())
                loader.set_postfix(accuracy_weather=weather_accuracy / len(data_loader))
    model.train()

torch.cuda.empty_cache()
root = "D:/autodrive/Perception/SceneClassifier/datasets"
if __name__ == '__main__':
    random_state = 42
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    annotation = open(os.path.join(root, 'train_dataset', 'train.json'))
    data = json.load(annotation)
    print('Total:', len(data['annotations']))

    train_ds = pd.json_normalize(data['annotations'])

    weather_encoder = LabelEncoder().fit(train_ds['weather'])
    train_ds['weather'] = weather_encoder.transform(train_ds['weather'])
    train_ds['filename'] = train_ds['filename'].str.replace('\\', '/', regex=True)
    print(train_ds['weather'].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(train_ds['filename'].values, train_ds['weather'].values,
                                                        shuffle=True,
                                                        random_state=random_state,
                                                        stratify=train_ds['weather'])

    transforms = {
        x: Compose([
            Resize(224, 224),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_REPLICATE),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            OneOf([
                GaussianBlur(),
                GaussNoise(),
            ], p=0.2),

            Normalize(),
            ToTensorV2()
        ]) if x == 'train' else Compose([
            Resize(224, 224),

            Normalize(),
            ToTensorV2()
        ]) for x in ['train', 'test']
    }

    sets = {'train': (x_train, y_train), 'test': (x_test, y_test)}
    datasets = {x: CustomDataset(sets[x][0], sets[x][1], transforms[x]) for x in sets.keys()}
    dataloaders = {x: DataLoader(datasets[x], batch_size=8, num_workers=2, pin_memory=True) for x in sets.keys()}
    model = MultiOutputEffNet(len(weather_encoder.classes_))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    epochs = 10
    for epoch in range(1, epochs + 1):
        train_fn(epoch, model, dataloaders['train'], loss_fn, optimizer, device)
        validation_fn(epoch, model, dataloaders['test'], device)

    torch.save(model.state_dict(), 'model.pt')
