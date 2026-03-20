"""VGG-style discriminator for GAN-based super-resolution training."""

import torch.nn as nn


class VGGStyleDiscriminator(nn.Module):
    """VGG-style discriminator with 128x128 input size.

    Args:
        num_in_ch: Number of input channels.
        num_feat: Number of base feature channels.
    """

    def __init__(self, num_in_ch=3, num_feat=64):
        super().__init__()

        self.features = nn.Sequential(
            # input: 128x128
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1),
            nn.BatchNorm2d(num_feat * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(num_feat * 8),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_feat * 8 * 8 * 8, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
