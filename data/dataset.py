"""Dataset for super-resolution training with old photo degradation."""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.degradation import OldPhotoDegradation
from utils.img_utils import img2tensor


class SRDataset(Dataset):
    """Super-resolution dataset that generates LR-HR pairs.

    Loads HR images and creates LR counterparts via downsampling
    and optional old photo degradation simulation.

    Args:
        data_dir: Directory containing HR images.
        hr_size: Size to crop HR images to (square).
        scale: Downsampling scale factor.
        use_flip: Enable random horizontal flip augmentation.
        use_rot: Enable random rotation augmentation.
        use_old_photo_degrade: Enable old photo degradation on LR images.
        degradation_config: Dict of degradation parameters.
    """

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, data_dir, hr_size=256, scale=4,
                 use_flip=True, use_rot=True,
                 use_old_photo_degrade=True, degradation_config=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.use_flip = use_flip
        self.use_rot = use_rot

        # Collect image paths
        self.image_paths = sorted([
            p for p in self.data_dir.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ])
        if not self.image_paths:
            raise RuntimeError(f'No images found in {data_dir}')

        # Old photo degradation
        self.degrader = None
        if use_old_photo_degrade:
            cfg = degradation_config or {}
            self.degrader = OldPhotoDegradation(**cfg)

    def __len__(self):
        return len(self.image_paths)

    def _random_crop(self, img, size):
        h, w = img.shape[:2]
        if h < size or w < size:
            img = cv2.resize(img, (max(w, size), max(h, size)),
                             interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return img[top:top + size, left:left + size]

    def _augment(self, img_hr, img_lr):
        # Random horizontal flip
        if self.use_flip and random.random() < 0.5:
            img_hr = np.flip(img_hr, axis=1).copy()
            img_lr = np.flip(img_lr, axis=1).copy()
        # Random rotation (0, 90, 180, 270)
        if self.use_rot:
            k = random.randint(0, 3)
            img_hr = np.rot90(img_hr, k).copy()
            img_lr = np.rot90(img_lr, k).copy()
        return img_hr, img_lr

    def __getitem__(self, idx):
        # Load HR image
        img_hr = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        if img_hr is None:
            # Fallback to next image
            return self.__getitem__((idx + 1) % len(self))

        # Random crop to HR size
        img_hr = self._random_crop(img_hr, self.hr_size)

        # Create LR image via downsampling
        img_lr = cv2.resize(
            img_hr, (self.lr_size, self.lr_size),
            interpolation=cv2.INTER_CUBIC
        )

        # Apply old photo degradation to LR
        if self.degrader is not None:
            img_lr = self.degrader(img_lr)

        # Data augmentation (applied consistently to both)
        img_hr, img_lr = self._augment(img_hr, img_lr)

        # Convert to tensors
        tensor_hr = img2tensor(img_hr)
        tensor_lr = img2tensor(img_lr)

        return {'lr': tensor_lr, 'hr': tensor_hr,
                'path': str(self.image_paths[idx])}


def create_dataloaders(config):
    """Create train and validation dataloaders from config dict.

    Args:
        config: Dict with 'train' and 'val' dataset settings.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_cfg = config['dataset']['train']
    val_cfg = config['dataset']['val']
    degrade_cfg = config.get('degradation', {})

    train_dataset = SRDataset(
        data_dir=train_cfg['data_dir'],
        hr_size=train_cfg['hr_size'],
        scale=train_cfg['scale'],
        use_flip=train_cfg.get('use_flip', True),
        use_rot=train_cfg.get('use_rot', True),
        use_old_photo_degrade=train_cfg.get('use_old_photo_degrade', True),
        degradation_config=degrade_cfg,
    )

    val_dataset = SRDataset(
        data_dir=val_cfg['data_dir'],
        hr_size=val_cfg['hr_size'],
        scale=val_cfg['scale'],
        use_flip=False,
        use_rot=False,
        use_old_photo_degrade=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=val_cfg.get('num_workers', 2),
        pin_memory=True,
    )

    return train_loader, val_loader
