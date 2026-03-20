"""Gradio GUI for Old Photo Super-Resolution.

Provides a web-based interface for uploading old photos and
viewing the super-resolution results side by side.

Usage:
    python app.py --model checkpoints/best_model.pth
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
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'generator' in state_dict:
        state_dict = state_dict['generator']
    _model.load_state_dict(state_dict)
    _model.eval()

    print(f'Model loaded from {model_path} on {_device}')


def super_resolve(input_image, tile_size=512):
    """Run super-resolution on an input image.

    Args:
        input_image: Input image (RGB, numpy array from Gradio).
        tile_size: Tile size for processing large images.

    Returns:
        Super-resolved image (RGB, numpy array).
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
    with gr.Blocks(
        title='Old Photo Super-Resolution',
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            '# Old Photo Super-Resolution\n'
            'Upload an old or low-resolution photo to enhance it using '
            'deep learning-based super-resolution (4x upscaling).'
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
                )

        run_btn.click(
            fn=super_resolve,
            inputs=[input_image, tile_slider],
            outputs=output_image,
        )

        gr.Markdown(
            '### Tips\n'
            '- For best results, use images smaller than 512x512 pixels\n'
            '- Reduce tile size if you encounter GPU memory errors\n'
            '- The model performs 4x upscaling\n'
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Old Photo SR GUI')
    parser.add_argument('--model', type=str,
                        default='checkpoints/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str,
                        default='configs/train_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio link')
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
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
