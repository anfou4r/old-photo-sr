"""Old photo restoration utilities.

Provides color correction, scratch/noise removal, and face enhancement
for old and degraded photographs.
"""

import cv2
import numpy as np


def auto_color_correction(img):
    """Fix color cast and fading in old photos.

    Uses white balance correction and histogram equalization
    to restore natural colors from yellowed/faded photos.

    Args:
        img: BGR uint8 numpy array.

    Returns:
        Color-corrected BGR uint8 numpy array.
    """
    # Convert to LAB for better color manipulation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # White balance: shift A and B channels to neutral
    l_ch, a_ch, b_ch = cv2.split(lab)
    a_mean = a_ch.mean()
    b_mean = b_ch.mean()
    # Move towards neutral (128) with 80% strength
    a_ch = np.clip(a_ch - (a_mean - 128) * 0.8, 0, 255)
    b_ch = np.clip(b_ch - (b_mean - 128) * 0.8, 0, 255)

    # CLAHE on L channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_uint8 = np.clip(l_ch, 0, 255).astype(np.uint8)
    l_eq = clahe.apply(l_uint8).astype(np.float32)

    lab_corrected = cv2.merge([l_eq, a_ch.astype(np.uint8).astype(np.float32),
                                b_ch.astype(np.uint8).astype(np.float32)])
    corrected = cv2.cvtColor(lab_corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return corrected


def remove_scratches(img, strength=50):
    """Remove scratches, fold marks, and spots from old photos.

    Uses morphological operations and inpainting to detect
    and remove linear artifacts (scratches, fold lines).

    Args:
        img: BGR uint8 numpy array.
        strength: Detection sensitivity (0-100). Higher = more aggressive.

    Returns:
        Cleaned BGR uint8 numpy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect scratches using morphological gradient
    # Scratches are thin bright/dark lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Detect vertical scratches
    close_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
    diff_v = cv2.absdiff(gray, close_h)

    # Detect horizontal scratches
    close_v = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v)
    diff_h = cv2.absdiff(gray, close_v)

    # Combine scratch masks
    scratch_mask = cv2.add(diff_v, diff_h)

    # Threshold based on strength
    thresh = max(10, 50 - strength // 2)
    _, mask = cv2.threshold(scratch_mask, thresh, 255, cv2.THRESH_BINARY)

    # Clean up mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel_clean, iterations=1)

    # Inpaint scratches
    result = cv2.inpaint(img, mask, inpaintRadius=3,
                         flags=cv2.INPAINT_TELEA)

    return result


def denoise_old_photo(img, strength=10):
    """Remove noise and grain from old photos.

    Args:
        img: BGR uint8 numpy array.
        strength: Denoising strength (1-30).

    Returns:
        Denoised BGR uint8 numpy array.
    """
    strength = max(1, min(30, strength))
    result = cv2.fastNlMeansDenoisingColored(
        img, None, strength, strength, 7, 21
    )
    return result


def restore_old_photo(img, color_fix=True, scratch_removal=True,
                      scratch_strength=50, denoise_strength=10,
                      face_enhance=True, face_enhancer=None):
    """Full old photo restoration pipeline.

    Pipeline order:
    1. Scratch/artifact removal
    2. Denoising
    3. Color correction
    4. Face enhancement (if available)

    Args:
        img: BGR uint8 numpy array.
        color_fix: Whether to apply color correction.
        scratch_removal: Whether to remove scratches.
        scratch_strength: Scratch detection sensitivity (0-100).
        denoise_strength: Denoising strength (0=off, 1-30).
        face_enhance: Whether to enhance faces.
        face_enhancer: GFPGANer instance (or None to skip).

    Returns:
        Restored BGR uint8 numpy array.
    """
    result = img.copy()

    # Step 1: Remove scratches
    if scratch_removal:
        result = remove_scratches(result, scratch_strength)

    # Step 2: Denoise
    if denoise_strength > 0:
        result = denoise_old_photo(result, denoise_strength)

    # Step 3: Color correction
    if color_fix:
        result = auto_color_correction(result)

    # Step 4: Face enhancement
    if face_enhance and face_enhancer is not None:
        _, _, enhanced = face_enhancer.enhance(
            result, has_aligned=False,
            only_center_face=False, paste_back=True
        )
        if enhanced is not None:
            result = enhanced

    return result
