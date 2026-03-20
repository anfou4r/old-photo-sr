"""Image I/O and conversion utilities."""

import cv2
import numpy as np
import torch


def load_image(path):
    """Load an image from disk as BGR uint8 numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {path}')
    return img


def save_image(img, path):
    """Save an image (BGR uint8 numpy array or tensor) to disk."""
    if isinstance(img, torch.Tensor):
        img = tensor2img(img)
    cv2.imwrite(str(path), img)


def img2tensor(img, bgr2rgb=True, float32=True):
    """Convert BGR uint8 numpy image to RGB float32 tensor (CHW, [0, 1])."""
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if not float32:
        img = img.half()
    return img


def tensor2img(tensor, rgb2bgr=True, min_max=(0, 1)):
    """Convert float tensor (CHW or NCHW) to BGR uint8 numpy image."""
    tensor = tensor.squeeze(0).detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img = tensor.numpy().transpose(1, 2, 0)
    img = (img * 255.0).round().astype(np.uint8)
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
