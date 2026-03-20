"""Loss functions for super-resolution training."""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG19 features.

    Computes L1 distance between feature maps of the generated
    and ground truth images at multiple VGG layers.
    """

    def __init__(self, layer_weights=None):
        super().__init__()
        if layer_weights is None:
            # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            self.layer_weights = {
                '2': 0.1, '7': 0.1, '16': 1.0, '25': 1.0, '34': 1.0
            }
        else:
            self.layer_weights = layer_weights

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleDict()
        max_idx = max(int(k) for k in self.layer_weights.keys())

        for i in range(max_idx + 1):
            self.slices[str(i)] = vgg[i]

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        pred_feat = pred
        target_feat = target

        for i in range(max(int(k) for k in self.layer_weights.keys()) + 1):
            pred_feat = self.slices[str(i)](pred_feat)
            target_feat = self.slices[str(i)](target_feat)
            if str(i) in self.layer_weights:
                loss += self.layer_weights[str(i)] * nn.functional.l1_loss(
                    pred_feat, target_feat
                )

        return loss


class GANLoss(nn.Module):
    """GAN loss with support for vanilla and relativistic variants.

    Args:
        gan_type: Type of GAN loss ('vanilla' or 'lsgan').
        real_label_val: Target value for real images.
        fake_label_val: Target value for fake images.
    """

    def __init__(self, gan_type='vanilla', real_label_val=1.0,
                 fake_label_val=0.0):
        super().__init__()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f'Unsupported GAN type: {gan_type}')

    def _get_target(self, pred, target_is_real):
        val = self.real_label_val if target_is_real else self.fake_label_val
        return torch.full_like(pred, val)

    def forward(self, pred, target_is_real):
        target = self._get_target(pred, target_is_real)
        return self.loss(pred, target)
