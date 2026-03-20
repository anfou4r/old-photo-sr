"""Image quality metrics for super-resolution evaluation."""

import numpy as np
import cv2


def calculate_psnr(img1, img2, crop_border=4):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        img1: Image 1 (uint8, BGR, HWC).
        img2: Image 2 (uint8, BGR, HWC).
        crop_border: Pixels to crop from borders before calculation.

    Returns:
        PSNR value in dB.
    """
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def calculate_ssim(img1, img2, crop_border=4):
    """Calculate SSIM (Structural Similarity Index).

    Args:
        img1: Image 1 (uint8, BGR, HWC).
        img2: Image 2 (uint8, BGR, HWC).
        crop_border: Pixels to crop from borders before calculation.

    Returns:
        SSIM value.
    """
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())
