"""Training script for Old Photo Super-Resolution.

Supports:
- PSNR-oriented pre-training (L1 loss only)
- GAN-based fine-tuning (L1 + Perceptual + GAN losses)
- Cosine annealing LR schedule with warmup
- TensorBoard logging
- Periodic validation and checkpoint saving

Usage:
    # PSNR pre-training
    python train.py --config configs/train_config.yaml --stage psnr

    # GAN fine-tuning (from pre-trained PSNR model)
    python train.py --config configs/train_config.yaml --stage gan \
        --pretrain checkpoints/best_psnr_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from models import RRDBNet, VGGStyleDiscriminator, PerceptualLoss, GANLoss
from data import create_dataloaders
from utils import calculate_psnr, calculate_ssim, tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Train Old Photo SR')
    parser.add_argument('--config', type=str,
                        default='configs/train_config.yaml')
    parser.add_argument('--stage', type=str, default='psnr',
                        choices=['psnr', 'gan'],
                        help='Training stage: psnr or gan')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Pre-trained generator weights for GAN stage')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    return parser.parse_args()


def cosine_lr_scheduler(optimizer, current_epoch, total_epochs,
                        warmup_epochs, base_lr):
    """Cosine annealing with linear warmup."""
    if current_epoch < warmup_epochs:
        lr = base_lr * (current_epoch + 1) / warmup_epochs
    else:
        import math
        progress = (current_epoch - warmup_epochs) / (
            total_epochs - warmup_epochs
        )
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def validate(model, val_loader, device):
    """Run validation and return average PSNR and SSIM."""
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            lr = batch['lr'].to(device)
            hr = batch['hr']

            sr = model(lr).cpu()

            for i in range(sr.size(0)):
                sr_img = tensor2img(sr[i])
                hr_img = tensor2img(hr[i])
                total_psnr += calculate_psnr(sr_img, hr_img)
                total_ssim += calculate_ssim(sr_img, hr_img)
                count += 1

    model.train()
    return total_psnr / max(count, 1), total_ssim / max(count, 1)


def train_psnr(config, device):
    """PSNR-oriented pre-training with L1 loss."""
    model_cfg = config['model']
    train_cfg = config['train']

    # Model
    net_g = RRDBNet(
        num_in_ch=model_cfg['num_in_ch'],
        num_out_ch=model_cfg['num_out_ch'],
        num_feat=model_cfg['num_feat'],
        num_block=model_cfg['num_block'],
        num_grow_ch=model_cfg['num_grow_ch'],
        scale=model_cfg['scale'],
    ).to(device)

    # Data
    train_loader, val_loader = create_dataloaders(config)

    # Optimizer
    optimizer_g = optim.Adam(
        net_g.parameters(),
        lr=train_cfg['lr_g'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )

    # Loss
    pixel_loss = nn.L1Loss().to(device)

    # Logging
    ckpt_dir = Path(train_cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(train_cfg['log_dir'])

    best_psnr = 0.0
    total_epochs = train_cfg['total_epochs']
    global_step = 0

    print(f'Starting PSNR pre-training for {total_epochs} epochs...')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    for epoch in range(total_epochs):
        net_g.train()
        lr = cosine_lr_scheduler(
            optimizer_g, epoch, total_epochs,
            train_cfg['warmup_epochs'], train_cfg['lr_g']
        )

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}')
        epoch_loss = 0.0

        for batch in pbar:
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)

            # Forward
            sr_img = net_g(lr_img)
            loss = pixel_loss(sr_img, hr_img)

            # Backward
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % train_cfg['log_interval'] == 0:
                writer.add_scalar('train/pixel_loss', loss.item(), global_step)
                writer.add_scalar('train/lr', lr, global_step)

            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.6f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}')

        # Validation
        if (epoch + 1) % train_cfg['val_interval'] == 0:
            psnr, ssim = validate(net_g, val_loader, device)
            writer.add_scalar('val/psnr', psnr, epoch + 1)
            writer.add_scalar('val/ssim', ssim, epoch + 1)
            print(f'  Validation - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(net_g.state_dict(),
                           ckpt_dir / 'best_psnr_model.pth')
                print(f'  Saved best model (PSNR: {psnr:.2f} dB)')

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net_g.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    writer.close()
    print(f'Training complete. Best PSNR: {best_psnr:.2f} dB')


def train_gan(config, pretrain_path, device):
    """GAN-based fine-tuning with perceptual and adversarial losses."""
    model_cfg = config['model']
    train_cfg = config['train']

    # Generator
    net_g = RRDBNet(
        num_in_ch=model_cfg['num_in_ch'],
        num_out_ch=model_cfg['num_out_ch'],
        num_feat=model_cfg['num_feat'],
        num_block=model_cfg['num_block'],
        num_grow_ch=model_cfg['num_grow_ch'],
        scale=model_cfg['scale'],
    ).to(device)

    # Load pre-trained weights
    if pretrain_path:
        state_dict = torch.load(pretrain_path, map_location=device,
                                weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        net_g.load_state_dict(state_dict)
        print(f'Loaded pre-trained generator from {pretrain_path}')

    # Discriminator
    net_d = VGGStyleDiscriminator(num_in_ch=3, num_feat=64).to(device)

    # Data
    train_loader, val_loader = create_dataloaders(config)

    # Optimizers
    optimizer_g = optim.Adam(
        net_g.parameters(), lr=train_cfg['lr_g'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )
    optimizer_d = optim.Adam(
        net_d.parameters(), lr=train_cfg['lr_d'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )

    # Losses
    pixel_loss = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    gan_loss = GANLoss(gan_type='vanilla').to(device)

    w_pixel = train_cfg['pixel_weight']
    w_percep = train_cfg['perceptual_weight']
    w_gan = train_cfg['gan_weight']

    # Logging
    ckpt_dir = Path(train_cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(train_cfg['log_dir'])

    best_psnr = 0.0
    total_epochs = train_cfg['total_epochs']
    global_step = 0

    print(f'Starting GAN fine-tuning for {total_epochs} epochs...')

    for epoch in range(total_epochs):
        net_g.train()
        net_d.train()
        lr = cosine_lr_scheduler(
            optimizer_g, epoch, total_epochs,
            train_cfg['warmup_epochs'], train_cfg['lr_g']
        )
        cosine_lr_scheduler(
            optimizer_d, epoch, total_epochs,
            train_cfg['warmup_epochs'], train_cfg['lr_d']
        )

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}')

        for batch in pbar:
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)

            sr_img = net_g(lr_img)

            # --- Update Discriminator ---
            for p in net_d.parameters():
                p.requires_grad = True

            # Crop to 128x128 for discriminator
            h = min(sr_img.size(2), 128)
            w = min(sr_img.size(3), 128)
            sr_crop = sr_img[:, :, :h, :w].detach()
            hr_crop = hr_img[:, :, :h, :w]

            pred_real = net_d(hr_crop)
            pred_fake = net_d(sr_crop)
            loss_d = (
                gan_loss(pred_real, True) + gan_loss(pred_fake, False)
            ) * 0.5

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # --- Update Generator ---
            for p in net_d.parameters():
                p.requires_grad = False

            sr_img = net_g(lr_img)
            sr_crop = sr_img[:, :, :h, :w]

            # Pixel loss
            l_pix = pixel_loss(sr_img, hr_img) * w_pixel
            # Perceptual loss
            l_percep = perceptual_loss(sr_img, hr_img) * w_percep
            # GAN loss
            pred_fake = net_d(sr_crop)
            l_gan = gan_loss(pred_fake, True) * w_gan

            loss_g = l_pix + l_percep + l_gan

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            global_step += 1
            if global_step % train_cfg['log_interval'] == 0:
                writer.add_scalar('train/loss_g', loss_g.item(), global_step)
                writer.add_scalar('train/loss_d', loss_d.item(), global_step)
                writer.add_scalar('train/l_pix', l_pix.item(), global_step)
                writer.add_scalar('train/l_percep', l_percep.item(),
                                  global_step)
                writer.add_scalar('train/l_gan', l_gan.item(), global_step)

            pbar.set_postfix(
                g=f'{loss_g.item():.4f}',
                d=f'{loss_d.item():.4f}',
            )

        # Validation
        if (epoch + 1) % train_cfg['val_interval'] == 0:
            psnr, ssim = validate(net_g, val_loader, device)
            writer.add_scalar('val/psnr', psnr, epoch + 1)
            writer.add_scalar('val/ssim', ssim, epoch + 1)
            print(f'  Validation - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(net_g.state_dict(), ckpt_dir / 'best_model.pth')

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator': net_g.state_dict(),
                'discriminator': net_d.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_dir / f'gan_checkpoint_epoch_{epoch + 1}.pth')

    writer.close()
    print(f'GAN training complete. Best PSNR: {best_psnr:.2f} dB')


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device: {device}')

    if args.stage == 'psnr':
        train_psnr(config, device)
    elif args.stage == 'gan':
        train_gan(config, args.pretrain, device)


if __name__ == '__main__':
    main()
