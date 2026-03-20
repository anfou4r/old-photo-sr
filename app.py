"""Gradio GUI for Old Photo Super-Resolution.

Provides a web-based interface for uploading old photos and
viewing the super-resolution results side by side, with optional
image enhancement post-processing.

Usage:
    python app.py --model checkpoints/best_model.pth
    python app.py --use_pretrained  # Auto-download Real-ESRGAN weights
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


# Global model reference
_model = None
_device = None

PRETRAINED_URL = (
    'https://github.com/xinntao/Real-ESRGAN/releases/download/'
    'v0.1.0/RealESRGAN_x4plus.pth'
)


def download_pretrained(save_path='checkpoints/RealESRGAN_x4plus.pth'):
    """Download Real-ESRGAN pre-trained weights."""
    save_path = Path(save_path)
    if save_path.exists():
        print(f'Pre-trained weights already exist at {save_path}')
        return str(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading Real-ESRGAN pre-trained weights...')
    print(f'URL: {PRETRAINED_URL}')

    import urllib.request
    urllib.request.urlretrieve(PRETRAINED_URL, str(save_path))
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
    # Support multiple checkpoint formats
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


def enhance_image(img, brightness=0, contrast=0, saturation=0,
                  sharpness=0, denoise=0):
    """Apply post-processing enhancements to the SR output.

    Args:
        img: Input image (RGB uint8 numpy array).
        brightness: Brightness adjustment (-50 to 50).
        contrast: Contrast adjustment (-50 to 50).
        saturation: Saturation adjustment (-50 to 50).
        sharpness: Sharpening strength (0 to 100).
        denoise: Denoising strength (0 to 30).

    Returns:
        Enhanced image (RGB uint8 numpy array).
    """
    result = img.copy()

    # Brightness and contrast
    if brightness != 0 or contrast != 0:
        alpha = 1.0 + contrast / 100.0  # contrast factor
        beta = brightness * 2.55  # brightness offset
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # Saturation
    if saturation != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = 1.0 + saturation / 50.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Denoising
    if denoise > 0:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.fastNlMeansDenoisingColored(
            result_bgr, None, denoise, denoise, 7, 21
        )
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Sharpening
    if sharpness > 0:
        strength = sharpness / 100.0
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.0 + strength, blurred,
                                 -strength, 0)

    return result


def super_resolve(input_image, tile_size=512, brightness=0, contrast=0,
                  saturation=0, sharpness=0, denoise=0):
    """Run super-resolution on an input image with optional enhancement.

    Args:
        input_image: Input image (RGB, numpy array from Gradio).
        tile_size: Tile size for processing large images.
        brightness: Brightness adjustment.
        contrast: Contrast adjustment.
        saturation: Saturation adjustment.
        sharpness: Sharpening strength.
        denoise: Denoising strength.

    Returns:
        Super-resolved and enhanced image (RGB, numpy array).
    """
    if _model is None:
        return input_image

    # Gradio gives RGB numpy array
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    img_tensor = img2tensor(img_bgr).unsqueeze(0)

    h, w = img_bgr.shape[:2]

    with torch.no_grad():
        if tile_size > 0 and (h > tile_size or w > tile_size):
            output = _tile_process(img_tensor, tile_size)
        else:
            output = _model(img_tensor.to(_device))

    # Convert back to RGB numpy
    output_bgr = tensor2img(output.float())
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    # Apply post-processing enhancements
    if any([brightness, contrast, saturation, sharpness, denoise]):
        output_rgb = enhance_image(
            output_rgb, brightness, contrast, saturation,
            sharpness, denoise
        )

    return output_rgb


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

            # Crop padding
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


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title='Old Photo Super-Resolution') as demo:
        gr.Markdown(
            '# Old Photo Super-Resolution\n'
            'Upload an old or low-resolution photo to enhance it using '
            'deep learning-based super-resolution (4x upscaling).\n'
            'Powered by Real-ESRGAN pre-trained model.'
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label='Input (Old/Low-res Photo)',
                    type='numpy',
                )
                tile_slider = gr.Slider(
                    minimum=0, maximum=1024, step=64, value=512,
                    label='Tile Size (0 = no tiling, reduce if OOM)',
                )
                run_btn = gr.Button('Enhance Photo', variant='primary')

            with gr.Column():
                output_image = gr.Image(
                    label='Output (Super-Resolved)',
                    type='numpy',
                    format='png',
                )

        with gr.Accordion('Image Enhancement (Post-processing)',
                          open=False):
            gr.Markdown(
                'Adjust these sliders to fine-tune the output. '
                'Click **Enhance Photo** again to apply.'
            )
            with gr.Row():
                brightness = gr.Slider(-50, 50, value=0, step=1,
                                       label='Brightness')
                contrast = gr.Slider(-50, 50, value=0, step=1,
                                     label='Contrast')
            with gr.Row():
                saturation = gr.Slider(-50, 50, value=0, step=1,
                                       label='Saturation')
                sharpness = gr.Slider(0, 100, value=0, step=1,
                                      label='Sharpness')
            with gr.Row():
                denoise = gr.Slider(0, 30, value=0, step=1,
                                    label='Denoise Strength')

        run_btn.click(
            fn=super_resolve,
            inputs=[input_image, tile_slider, brightness, contrast,
                    saturation, sharpness, denoise],
            outputs=output_image,
        )

        gr.Markdown(
            '### Tips\n'
            '- For best results, use images smaller than 512x512 pixels\n'
            '- Reduce tile size if you encounter GPU memory errors\n'
            '- The model performs 4x upscaling\n'
            '- Use the enhancement sliders to adjust brightness, '
            'contrast, and sharpness after SR\n'
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Old Photo SR GUI')
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
                        help='Download and use Real-ESRGAN pretrained model')
    args = parser.parse_args()

    # Auto-download pretrained weights if needed
    model_path = Path(args.model)
    if not model_path.exists() and args.use_pretrained:
        args.model = download_pretrained()
        model_path = Path(args.model)

    # Load model
    if model_path.exists():
        load_model(args.model, args.config, args.gpu)
    else:
        print(f'Warning: Model not found at {args.model}. '
              'GUI will start without a model.')

    # Launch UI
    demo = build_ui()
    demo.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
