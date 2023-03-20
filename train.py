import json

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import random_split, DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import transforms

from PIL import Image

from torchsummary import summary

from models import UNet
from utils import *

CONFIG = load_config("./configs/base.yml", verbose=True)

transform = transforms.Compose([
              transforms.Resize(CONFIG['img_size']),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), 
                                   (0.229, 0.224, 0.225))  
])

def target_transform(target: Image):
    img = target.convert("RGB")

    transform = transforms.Compose([
              transforms.Resize(CONFIG['img_size']),
              transforms.ToTensor()])
    
    img = transform(img)
    return img

trainval_set = OxfordIIITPet("./datasets", "trainval", 
                             target_types="segmentation", 
                             transform=transform, 
                             target_transform=target_transform, 
                             download=True)

test_set = OxfordIIITPet("./datasets", "test", 
                        target_types="segmentation", 
                        transform=transform, 
                        target_transform=target_transform, 
                        download=True)


train_set, val_set = random_split(trainval_set, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=CONFIG['hyperparams']['batch_size'])
val_loader = DataLoader(val_set, batch_size=CONFIG['hyperparams']['batch_size'])
test_loader = DataLoader(test_set, batch_size=CONFIG['hyperparams']['batch_size'])

model = UNet(out_channels=3).to(CONFIG['device'])

summary(model, (3, 256, 256), verbose=1)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=CONFIG['hyperparams']['lr'])
early_stopping = EarlyStopping(save_path=CONFIG['model_save'],
                               patience=CONFIG['early_stopping']['patience'],
                               delta=CONFIG['early_stopping']['delta'],
                               verbose=CONFIG['early_stopping']['verbose'])

history = train(model=model,
                dataloaders={'train': train_loader,
                             'val': val_loader},
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=CONFIG['hyperparams']['num_epochs'],
                early_stopping=early_stopping,
                device=CONFIG['device'])

if CONFIG['training_state_dict']:
    with open(CONFIG['train_state_dict'], "w", encoding='utf8') as f:
        json.dump(history, f, indent=4)