"""Gradio GUI for Old Photo Super-Resolution & Restoration.

Provides a web-based interface for uploading old photos and
performing super-resolution, color correction, scratch removal,
and face enhancement.

Usage:
    python app.py --use_pretrained --port 6006
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import gradio as gr

from models import RRDBNet
from utils import img2tensor, tensor2img
from restoration import (
    auto_color_correction, remove_scratches,
    denoise_old_photo, restore_old_photo,
)


# Global model references
_model = None
_device = None
_face_enhancer = None

PRETRAINED_URL = (
    'https://github.com/xinntao/Real-ESRGAN/releases/download/'
    'v0.1.0/RealESRGAN_x4plus.pth'
)

GFPGAN_URL = (
    'https://github.com/TencentARC/GFPGAN/releases/download/'
    'v1.3.0/GFPGANv1.3.pth'
)


def download_file(url, save_path):
    """Download a file from URL if not already present."""
    save_path = Path(save_path)
    if save_path.exists():
        print(f'Already exists: {save_path}')
        return str(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}...')
    import urllib.request
    urllib.request.urlretrieve(url, str(save_path))
    print(f'Downloaded to {save_path}')
    return str(save_path)


def load_model(model_path, config_path='configs/train_config.yaml',
               gpu=0):
    """Load the SR model."""
    global _model, _device

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']

    if gpu >= 0 and torch.cuda.is_available():
        _device = torch.device(f'cuda:{gpu}')
    else:
        _device = torch.device('cpu')

    _model = RRDBNet(
        num_in_ch=model_cfg['num_in_ch'],
        num_out_ch=model_cfg['num_out_ch'],
        num_feat=model_cfg['num_feat'],
        num_block=model_cfg['num_block'],
        num_grow_ch=model_cfg['num_grow_ch'],
        scale=model_cfg['scale'],
    ).to(_device)

    state_dict = torch.load(model_path, map_location=_device,
                            weights_only=True)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'generator' in state_dict:
        state_dict = state_dict['generator']
    _model.load_state_dict(state_dict, strict=True)
    _model.eval()

    print(f'Model loaded from {model_path} on {_device}')


def load_face_enhancer(model_path='checkpoints/GFPGANv1.3.pth'):
    """Load GFPGAN face enhancement model."""
    global _face_enhancer
    try:
        from gfpgan import GFPGANer
        _face_enhancer = GFPGANer(
            model_path=model_path,
            upscale=1,  # SR is handled separately
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
        print(f'GFPGAN loaded from {model_path}')
    except ImportError:
        print('Warning: gfpgan not installed. Face enhancement disabled.')
        print('Install with: pip install gfpgan')
    except Exception as e:
        print(f'Warning: Failed to load GFPGAN: {e}')


def _tile_process(img_tensor, tile_size, pad=32):
    """Process large images in tiles."""
    scale = 4
    _, _, h, w = img_tensor.size()
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, device=_device)
    counts = torch.zeros(1, 1, out_h, out_w, device=_device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(h, y + tile_size + pad)
            x1 = min(w, x + tile_size + pad)

            tile = img_tensor[:, :, y0:y1, x0:x1].to(_device)
            with torch.no_grad():
                tile_out = _model(tile)

            cy0 = (y - y0) * scale
            cx0 = (x - x0) * scale
            cy1 = tile_out.size(2) - (y1 - min(h, y + tile_size)) * scale
            cx1 = tile_out.size(3) - (x1 - min(w, x + tile_size)) * scale

            oy0 = y * scale
            ox0 = x * scale
            oy1 = min(out_h, (y + tile_size) * scale)
            ox1 = min(out_w, (x + tile_size) * scale)

            crop = tile_out[:, :, cy0:cy1, cx0:cx1]
            th = oy1 - oy0
            tw = ox1 - ox0
            output[:, :, oy0:oy1, ox0:ox1] += crop[:, :, :th, :tw]
            counts[:, :, oy0:oy1, ox0:ox1] += 1

    return output / counts.clamp(min=1)


def super_resolve(input_image, tile_size=512):
    """Run super-resolution on an input image.

    Args:
        input_image: RGB numpy array from Gradio.
        tile_size: Tile size for processing large images.

    Returns:
        Super-resolved RGB numpy array.
    """
    if _model is None or input_image is None:
        return input_image

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    img_tensor = img2tensor(img_bgr).unsqueeze(0)

    h, w = img_bgr.shape[:2]

    with torch.no_grad():
        if tile_size > 0 and (h > tile_size or w > tile_size):
            output = _tile_process(img_tensor, tile_size)
        else:
            output = _model(img_tensor.to(_device))

    output_bgr = tensor2img(output.float())
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    return output_rgb


def full_restore(input_image, tile_size=512,
                 enable_sr=True, enable_color_fix=True,
                 enable_scratch_removal=False, scratch_strength=50,
                 enable_denoise=True, denoise_strength=10,
                 enable_face=True,
                 brightness=0, contrast=0, saturation=0, sharpness=0):
    """Full old photo restoration pipeline.

    Args:
        input_image: RGB numpy array from Gradio.
        tile_size: Tile size for SR processing.
        enable_sr: Whether to run super-resolution.
        enable_color_fix: Whether to fix colors.
        enable_scratch_removal: Whether to remove scratches.
        scratch_strength: Scratch detection sensitivity.
        enable_denoise: Whether to denoise.
        denoise_strength: Denoising strength.
        enable_face: Whether to enhance faces.
        brightness/contrast/saturation/sharpness: Post-processing adjustments.

    Returns:
        Restored RGB numpy array.
    """
    if input_image is None:
        return None

    # Convert to BGR for processing
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Step 1: Pre-SR restoration (scratches, denoise, color on original)
    img_bgr = restore_old_photo(
        img_bgr,
        color_fix=enable_color_fix,
        scratch_removal=enable_scratch_removal,
        scratch_strength=scratch_strength,
        denoise_strength=denoise_strength if enable_denoise else 0,
        face_enhance=False,
        face_enhancer=None,
    )

    # Step 2: Super-resolution
    if enable_sr and _model is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = super_resolve(img_rgb, tile_size)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Step 3: Face enhancement (after SR for best results)
    if enable_face and _face_enhancer is not None:
        _, _, enhanced = _face_enhancer.enhance(
            img_bgr, has_aligned=False,
            only_center_face=False, paste_back=True
        )
        if enhanced is not None:
            img_bgr = enhanced

    # Step 4: Post-processing adjustments
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if brightness != 0 or contrast != 0:
        alpha = 1.0 + contrast / 100.0
        beta = brightness * 2.55
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    if saturation != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = 1.0 + saturation / 50.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if sharpness > 0:
        strength = sharpness / 100.0
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.0 + strength, blurred,
                                 -strength, 0)

    return result


def build_ui():
    """Build the Gradio interface."""
    face_status = 'Available' if _face_enhancer else 'Not available (install gfpgan)'

    with gr.Blocks(title='Old Photo Restoration') as demo:
        gr.Markdown(
            '# Old Photo Restoration & Super-Resolution\n'
            'Upload an old or low-resolution photo for comprehensive restoration:\n'
            '**4x Super-Resolution** | **Color Correction** | '
            '**Scratch Removal** | **Face Enhancement**\n\n'
            f'Face Enhancement: {face_status}'
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label='Input (Old/Low-res Photo)',
                    type='numpy',
                )

            with gr.Column():
                output_image = gr.Image(
                    label='Output (Restored)',
                    type='numpy',
                    format='png',
                )

        with gr.Row():
            run_btn = gr.Button('Restore Photo', variant='primary',
                                size='lg')

        with gr.Accordion('Restoration Options', open=True):
            with gr.Row():
                enable_sr = gr.Checkbox(value=True,
                                        label='Super-Resolution (4x)')
                enable_color_fix = gr.Checkbox(value=True,
                                                label='Auto Color Correction')
                enable_face = gr.Checkbox(
                    value=_face_enhancer is not None,
                    label='Face Enhancement (GFPGAN)',
                    interactive=_face_enhancer is not None,
                )

            with gr.Row():
                enable_scratch = gr.Checkbox(value=True,
                                              label='Scratch Removal')
                scratch_strength = gr.Slider(
                    0, 100, value=50, step=5,
                    label='Scratch Detection Sensitivity',
                )

            with gr.Row():
                enable_denoise = gr.Checkbox(value=True,
                                              label='Denoise')
                denoise_strength = gr.Slider(
                    1, 30, value=5, step=1,
                    label='Denoise Strength',
                )

            with gr.Row():
                tile_slider = gr.Slider(
                    minimum=0, maximum=1024, step=64, value=512,
                    label='Tile Size (0 = no tiling, reduce if OOM)',
                )

        with gr.Accordion('Fine-tune Adjustments', open=False):
            with gr.Row():
                brightness = gr.Slider(-50, 50, value=0, step=1,
                                       label='Brightness')
                contrast = gr.Slider(-50, 50, value=0, step=1,
                                     label='Contrast')
            with gr.Row():
                saturation = gr.Slider(-50, 50, value=0, step=1,
                                       label='Saturation')
                sharpness = gr.Slider(0, 100, value=30, step=1,
                                      label='Sharpness')

        run_btn.click(
            fn=full_restore,
            inputs=[input_image, tile_slider,
                    enable_sr, enable_color_fix,
                    enable_scratch, scratch_strength,
                    enable_denoise, denoise_strength,
                    enable_face,
                    brightness, contrast, saturation, sharpness],
            outputs=output_image,
        )

        gr.Markdown(
            '### Tips\n'
            '- **Color Correction** fixes yellowing and fading\n'
            '- **Scratch Removal** detects and removes lines/fold marks '
            '(enable only if photo has visible scratches)\n'
            '- **Face Enhancement** sharpens faces using GFPGAN\n'
            '- Reduce **Tile Size** if you get GPU memory errors\n'
            '- Use **Fine-tune Adjustments** for final touch-ups\n'
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Old Photo Restoration')
    parser.add_argument('--model', type=str,
                        default='checkpoints/RealESRGAN_x4plus.pth',
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str,
                        default='configs/train_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio link')
    parser.add_argument('--use_pretrained', action='store_true',
                        default=True,
                        help='Download and use pretrained models')
    parser.add_argument('--no_face', action='store_true',
                        help='Disable face enhancement')
    args = parser.parse_args()

    # Download pretrained weights
    model_path = Path(args.model)
    if not model_path.exists() and args.use_pretrained:
        args.model = download_file(PRETRAINED_URL, args.model)
        model_path = Path(args.model)

    # Load SR model
    if model_path.exists():
        load_model(args.model, args.config, args.gpu)

    # Load GFPGAN face enhancer
    if not args.no_face:
        gfpgan_path = 'checkpoints/GFPGANv1.3.pth'
        if not Path(gfpgan_path).exists() and args.use_pretrained:
            download_file(GFPGAN_URL, gfpgan_path)
        if Path(gfpgan_path).exists():
            load_face_enhancer(gfpgan_path)

    # Launch UI
    demo = build_ui()
    demo.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
