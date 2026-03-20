"""Dataset preparation script.

Downloads and prepares public datasets for training the super-resolution model.
Supports DIV2K and Flickr2K datasets commonly used in SR research.

Usage:
    python prepare_data.py --dataset div2k --output datasets/
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def split_dataset(data_dir, train_ratio=0.9):
    """Split images into train/val sets.

    Args:
        data_dir: Directory containing all images.
        train_ratio: Fraction of images for training.
    """
    data_dir = Path(data_dir)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    all_images = sorted([
        p for p in data_dir.rglob('*')
        if p.suffix.lower() in extensions
    ])

    if not all_images:
        print(f'No images found in {data_dir}')
        return

    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    split = int(len(all_images) * train_ratio)

    train_dir = data_dir.parent / 'train'
    val_dir = data_dir.parent / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f'Splitting {len(all_images)} images: '
          f'{split} train, {len(all_images) - split} val')

    for i, idx in enumerate(tqdm(indices, desc='Copying files')):
        src = all_images[idx]
        dst_dir = train_dir if i < split else val_dir
        shutil.copy2(src, dst_dir / src.name)

    print(f'Train images: {len(list(train_dir.iterdir()))}')
    print(f'Val images: {len(list(val_dir.iterdir()))}')


def create_patches(input_dir, output_dir, patch_size=512, stride=256):
    """Crop images into patches for more efficient training.

    Args:
        input_dir: Directory containing full-size images.
        output_dir: Output directory for patches.
        patch_size: Size of each patch.
        stride: Stride between patches.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    patch_count = 0
    for img_path in tqdm(images, desc='Creating patches'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y:y + patch_size, x:x + patch_size]
                patch_name = f'{img_path.stem}_p{patch_count:06d}.png'
                cv2.imwrite(str(output_dir / patch_name), patch)
                patch_count += 1

    print(f'Created {patch_count} patches in {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Prepare SR dataset')
    subparsers = parser.add_subparsers(dest='command')

    # Split command
    split_parser = subparsers.add_parser('split',
                                         help='Split dataset into train/val')
    split_parser.add_argument('--input', type=str, required=True,
                              help='Directory containing all images')
    split_parser.add_argument('--ratio', type=float, default=0.9,
                              help='Train split ratio')

    # Patch command
    patch_parser = subparsers.add_parser('patch',
                                         help='Create image patches')
    patch_parser.add_argument('--input', type=str, required=True)
    patch_parser.add_argument('--output', type=str, required=True)
    patch_parser.add_argument('--size', type=int, default=512)
    patch_parser.add_argument('--stride', type=int, default=256)

    args = parser.parse_args()

    if args.command == 'split':
        split_dataset(args.input, args.ratio)
    elif args.command == 'patch':
        create_patches(args.input, args.output, args.size, args.stride)
    else:
        parser.print_help()
        print('\nExample usage:')
        print('  # Split images into train/val:')
        print('  python prepare_data.py split --input datasets/all_images')
        print()
        print('  # Create patches from images:')
        print('  python prepare_data.py patch --input datasets/train '
              '--output datasets/train_patches --size 512 --stride 256')
        print()
        print('Recommended public datasets for super-resolution:')
        print('  - DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/')
        print('  - Flickr2K: https://cv.snu.ac.kr/research/EDSR/')
        print('  - Set5/Set14/BSD100: Standard SR benchmarks')
        print()
        print('Download HR images and place them in datasets/ directory,')
        print('then use the split command to create train/val splits.')


if __name__ == '__main__':
    main()
