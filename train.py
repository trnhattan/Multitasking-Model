import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import random_split, DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import transforms

from PIL import Image
from tqdm import tqdm

from torchsummary import summary

from models import UNet
from utils import *

CONFIG = {
    'num_epochs': 10,
    'batch_size': 12,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

img_size = (256, 256)

transform = transforms.Compose([
              transforms.Resize(img_size),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), 
                                   (0.229, 0.224, 0.225))  
])

def target_transform(target: Image):
    img = target.convert("RGB")

    transform = transforms.Compose([
              transforms.Resize(img_size),
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

train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'])
val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'])
test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'])

model = UNet(out_channels=3).to(CONFIG['device'])

summary(model, (3, 256, 256), verbose=1)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])

#@markdown `train(model, dataloaders, criterion, optimizer, num_epochs, device='cuda')`
def train(model, dataloaders, criterion, optimizer, num_epochs, device='cuda'):

    LOGGER.info(f"{emojis('✅')} {colorstr('Device:')} {device}")
    LOGGER.info(f"{emojis('✅')} {colorstr('Optimizer:')} {optimizer}")

    LOGGER.info(
        f"\n{emojis('✅')} {colorstr('Loss:')} {type(criterion).__name__}")

    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Loop over the data for each epoch
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs - 1}:'))
        LOGGER.info(colorstr('-' * 10))
        for phase in ['train', 'val']:
            running_loss = 0.0

            if phase == 'train':
                LOGGER.info(colorstr('black', 'bold', '%20s' + '%15s' * 2) % 
                            ('Training:', 'gpu_mem', 'loss'))
                model.train()
            else:
                LOGGER.info(colorstr('black', 'bold', '\n%20s' + '%15s' * 2) % 
                            ('Validation:','gpu_mem', 'loss'))
                model.eval()

            _phase = tqdm(dataloaders[phase],
                      total=len(dataloaders[phase]),
                      bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                      unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 1) % (mem, running_loss)
                _phase.set_description_str(desc)

            history[f'{phase}_loss'].append(running_loss)

    return history


train(
    model=model,
    dataloaders={
        'train': train_loader,
        'val': val_loader
    },
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=CONFIG['num_epochs'],
    device=CONFIG['device']
)