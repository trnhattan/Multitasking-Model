from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm

from utils import LOGGER, emojis, colorstr

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path: str, patience: int = 5, delta: float = 0, verbose: bool = False):
        """

        Args:
            patience (int, optional): How long to wait after last time validation loss improved. 
                (default: `7`)
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. 
                (default: `0`)
            save_path (str, optional): Path for the checkpoint to be saved to. 
                (default: `checkpoint.pt`)
            verbose (bool, optional): If True, prints a message for each validation loss improvement. 
                (default: `False`)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor_min = float('inf')
        self.delta = delta
        self.save_path = save_path

    def __call__(self, model, monitor):
        """Early Stopping spectator

        Args:
            model (any): trained model
            monitor (any): Monitored quantity
        """

        score = -monitor

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, monitor)
        elif score < self.best_score + self.delta:
            self.counter += 1
            LOGGER.info(colorstr('black', 'bold', f'EarlyStopping counter {self.counter} out of {self.patience}'))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, monitor)
            self.counter = 0

    def save_checkpoint(self, model, monitor):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            LOGGER.info(colorstr('black', 'bold', f'Validation loss decreased from {self.monitor_min:.6f} to {monitor:.6f}'))
        torch.save(model.state_dict(), self.save_path)
        self.monitor_min = monitor

def train(model, 
          dataloaders: dict, 
          criterion, 
          optimizer, 
          num_epochs: int,
          scheduler: _LRScheduler = None,
          early_stopping: Optional[EarlyStopping] = None,
          device='cuda'):
    """ `Train` function

    Args:
        dataloaders (dict): a dictionary with 'train' and/or 'val' keys
            that contains train `DataLoader` set and optional validation `DataLoader` set.
            Examples:
                >>> dataloaders = {
                ...     'train': train_loader (torch.utils.data.Dataset)
                ...     'val': val_loader (torch.utils.data.Dataset)
                ... }

        criterion (torch.nn): loss function
        optimizer (Optimizer): optimizer
        num_epochs (int): number of training epochs
        scheduler (_LRScheduler, optional): learning rate scheduler
            (default: `None`)
        early_stopping (EarlyStopping, optional): Early Stopping.
            (Defaults: `None`)
        device (str, optional): cpu or cuda.
            (Defaults: `'cuda'`)
        
    """

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


            if phase == 'val' and early_stopping:
                early_stopping(model, running_loss)
                if early_stopping.early_stop:                
                    LOGGER.info(f"{emojis('✅')} {colorstr(f'Early stopping at epoch {epoch}')}")
                    break

    return history