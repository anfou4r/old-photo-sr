"""Inference script for Old Photo Super-Resolution.

Supports single image and batch processing with optional tiled inference
for large images to avoid GPU memory issues.

Usage:
    python inference.py --input inputs/ --output results/ \
        --model checkpoints/best_model.pth
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models import RRDBNet
from utils import load_image, save_image, img2tensor, tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Old Photo SR Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--model', type=str,
                        default='checkpoints/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str,
                        default='configs/train_config.yaml',
                        help='Config file for model architecture')
    parser.add_argument('--scale', type=int, default=4,
                        help='Upscaling factor')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Tile size for large images (0 = no tiling)')
    parser.add_argument('--tile_pad', type=int, default=32,
                        help='Padding for tile boundaries')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id (-1 for CPU)')
    parser.add_argument('--half', action='store_true',
                        help='Use fp16 inference')
    return parser.parse_args()


def tile_inference(model, img_tensor, tile_size, tile_pad, scale, device):
    """Process a large image in overlapping tiles to save memory.

    Args:
        model: Super-resolution model.
        img_tensor: Input tensor (1, C, H, W).
        tile_size: Size of each tile.
        tile_pad: Padding overlap between tiles.
        scale: Upscaling factor.
        device: Torch device.

    Returns:
        Output tensor (1, C, H*scale, W*scale).
    """
    _, _, h, w = img_tensor.size()
    output_h, output_w = h * scale, w * scale
    output = torch.zeros(1, 3, output_h, output_w, device=device)
    weight_map = torch.zeros(1, 1, output_h, output_w, device=device)

    # Calculate tile positions
    tiles_y = list(range(0, h, tile_size))
    tiles_x = list(range(0, w, tile_size))

    for y in tiles_y:
        for x in tiles_x:
            # Input tile with padding
            y_start = max(0, y - tile_pad)
            x_start = max(0, x - tile_pad)
            y_end = min(h, y + tile_size + tile_pad)
            x_end = min(w, x + tile_size + tile_pad)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                tile_output = model(tile.to(device))

            # Remove padding from output
            out_y_start = (y - y_start) * scale
            out_x_start = (x - x_start) * scale
            out_y_end = tile_output.size(2) - (y_end - min(h, y + tile_size)) * scale
            out_x_end = tile_output.size(3) - (x_end - min(w, x + tile_size)) * scale

            # Position in output
            oy_start = y * scale
            ox_start = x * scale
            oy_end = min(output_h, (y + tile_size) * scale)
            ox_end = min(output_w, (x + tile_size) * scale)

            crop = tile_output[:, :, out_y_start:out_y_end,
                               out_x_start:out_x_end]
            target_h = oy_end - oy_start
            target_w = ox_end - ox_start
            output[:, :, oy_start:oy_end, ox_start:ox_end] += \
                crop[:, :, :target_h, :target_w]
            weight_map[:, :, oy_start:oy_end, ox_start:ox_end] += 1

    output /= weight_map.clamp(min=1)
    return output


def process_image(model, img_path, output_dir, args, device):
    """Process a single image through the SR model."""
    img = load_image(img_path)
    img_tensor = img2tensor(img).unsqueeze(0)

    if args.half:
        img_tensor = img_tensor.half()

    start_time = time.time()

    if args.tile_size > 0 and (img.shape[0] > args.tile_size
                               or img.shape[1] > args.tile_size):
        output = tile_inference(
            model, img_tensor, args.tile_size,
            args.tile_pad, args.scale, device
        )
    else:
        with torch.no_grad():
            output = model(img_tensor.to(device))

    elapsed = time.time() - start_time

    # Save result
    output_img = tensor2img(output.float())
    output_path = output_dir / f'{Path(img_path).stem}_sr.png'
    save_image(output_img, output_path)

    h, w = img.shape[:2]
    print(f'  {Path(img_path).name}: {w}x{h} -> '
          f'{w * args.scale}x{h * args.scale} ({elapsed:.2f}s)')

    return output_path


def main():
    args = parse_args()

    # Load config for model architecture
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Build model
    model = RRDBNet(
        num_in_ch=model_cfg['num_in_ch'],
        num_out_ch=model_cfg['num_out_ch'],
        num_feat=model_cfg['num_feat'],
        num_block=model_cfg['num_block'],
        num_grow_ch=model_cfg['num_grow_ch'],
        scale=model_cfg['scale'],
    ).to(device)

    # Load weights
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'generator' in state_dict:
        state_dict = state_dict['generator']
    model.load_state_dict(state_dict)
    model.eval()

    if args.half:
        model = model.half()

    print(f'Loaded model from {args.model}')

    # Prepare I/O
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in extensions
        ])

    print(f'Processing {len(image_paths)} image(s)...')
    for img_path in image_paths:
        process_image(model, img_path, output_dir, args, device)

    print('Done!')


if __name__ == '__main__':
    main()
