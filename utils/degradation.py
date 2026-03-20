"""Old photo degradation simulation.

Simulates various degradation artifacts commonly found in old photographs:
- Noise (Gaussian)
- Blur
- JPEG compression artifacts
- Color fading / sepia tone
- Scratches
- Vignette effect
"""

import cv2
import numpy as np
import random


class OldPhotoDegradation:
    """Applies realistic old photo degradation to images.

    Args:
        noise_range: Range of Gaussian noise sigma [min, max].
        jpeg_range: Range of JPEG quality [min, max].
        blur_kernel_range: Range of blur kernel sizes [min, max].
        blur_sigma_range: Range of Gaussian blur sigma [min, max].
        color_jitter: Whether to apply color jitter.
        sepia_prob: Probability of applying sepia tone.
        scratch_prob: Probability of adding scratches.
        vignette_prob: Probability of adding vignette.
    """

    def __init__(self, noise_range=(5, 30), jpeg_range=(30, 85),
                 blur_kernel_range=(3, 7), blur_sigma_range=(0.5, 3.0),
                 color_jitter=True, sepia_prob=0.3,
                 scratch_prob=0.2, vignette_prob=0.2):
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        self.color_jitter = color_jitter
        self.sepia_prob = sepia_prob
        self.scratch_prob = scratch_prob
        self.vignette_prob = vignette_prob

    def add_gaussian_noise(self, img):
        sigma = random.uniform(*self.noise_range)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def add_blur(self, img):
        ksize = random.randrange(
            self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2
        )
        sigma = random.uniform(*self.blur_sigma_range)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def add_jpeg_compression(self, img):
        quality = random.randint(*self.jpeg_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    def add_sepia(self, img):
        """Apply sepia/yellowish tone typical of aged photos."""
        sepia_filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_img = cv2.transform(img, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        # Blend with original
        alpha = random.uniform(0.3, 0.8)
        blended = cv2.addWeighted(img, 1 - alpha, sepia_img, alpha, 0)
        return blended

    def add_scratches(self, img):
        """Add random scratches to simulate old photo damage."""
        h, w = img.shape[:2]
        result = img.copy()
        num_scratches = random.randint(2, 8)
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            color = random.randint(180, 255)
            thickness = random.randint(1, 2)
            cv2.line(result, (x1, y1), (x2, y2),
                     (color, color, color), thickness)
        return result

    def add_vignette(self, img):
        """Add vignette (darkened edges) effect."""
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1.0 - (dist / max_dist) * random.uniform(0.3, 0.6)
        vignette = np.clip(vignette, 0, 1)
        result = (img.astype(np.float32) * vignette[..., np.newaxis])
        return np.clip(result, 0, 255).astype(np.uint8)

    def add_color_fade(self, img):
        """Reduce saturation to simulate faded colors."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.3, 0.7)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __call__(self, img):
        """Apply random degradation pipeline to an image.

        Args:
            img: BGR uint8 image (H, W, 3).

        Returns:
            Degraded BGR uint8 image.
        """
        # Always apply: blur, noise, jpeg
        if random.random() < 0.8:
            img = self.add_blur(img)
        if random.random() < 0.8:
            img = self.add_gaussian_noise(img)

        # Color effects
        if self.color_jitter and random.random() < 0.5:
            img = self.add_color_fade(img)
        if random.random() < self.sepia_prob:
            img = self.add_sepia(img)

        # Physical damage
        if random.random() < self.scratch_prob:
            img = self.add_scratches(img)
        if random.random() < self.vignette_prob:
            img = self.add_vignette(img)

        # JPEG compression (always last)
        if random.random() < 0.7:
            img = self.add_jpeg_compression(img)

        return img
