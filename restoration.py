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

    Uses multi-scale morphological detection and adaptive inpainting
    to detect and remove linear artifacts (scratches, fold lines, cracks).

    Args:
        img: BGR uint8 numpy array.
        strength: Detection sensitivity (0-100). Higher = more aggressive.

    Returns:
        Cleaned BGR uint8 numpy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multi-scale scratch detection for different scratch widths
    combined_mask = np.zeros_like(gray)

    # Scale 1: Fine scratches (thin lines)
    for length in [15, 25, 35]:
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

        close_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
        diff_v = cv2.absdiff(gray, close_h)

        close_v = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v)
        diff_h = cv2.absdiff(gray, close_v)

        combined_mask = cv2.add(combined_mask, cv2.add(diff_v, diff_h))

    # Scale 2: Diagonal scratches using rotated kernels
    for angle in [45, 135]:
        k = np.zeros((15, 15), dtype=np.uint8)
        if angle == 45:
            for i in range(15):
                k[i, i] = 1
        else:
            for i in range(15):
                k[i, 14 - i] = 1
        close_d = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
        diff_d = cv2.absdiff(gray, close_d)
        combined_mask = cv2.add(combined_mask, diff_d)

    # Normalize and enhance scratch mask
    combined_mask = cv2.normalize(combined_mask, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive threshold based on strength
    thresh = max(5, 45 - int(strength * 0.4))
    _, mask = cv2.threshold(combined_mask, thresh, 255, cv2.THRESH_BINARY)

    # Also detect dark spots/damage using local statistics
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    local_diff = cv2.absdiff(gray, blur)
    spot_thresh = max(15, 40 - int(strength * 0.3))
    _, spot_mask = cv2.threshold(local_diff, spot_thresh, 255,
                                 cv2.THRESH_BINARY)

    # Keep only small connected components (spots, not large regions)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        spot_mask, connectivity=8)
    max_area = img.shape[0] * img.shape[1] * 0.002  # Max 0.2% of image
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            spot_mask[labels == i] = 0

    # Combine scratch and spot masks
    mask = cv2.bitwise_or(mask, spot_mask)

    # Clean up: dilate slightly to cover scratch edges
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel_clean, iterations=1)

    # Remove small noise from mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Use NS inpainting (better for scratches) with adaptive radius
    inpaint_radius = max(3, min(7, 3 + strength // 25))
    result = cv2.inpaint(img, mask, inpaintRadius=inpaint_radius,
                         flags=cv2.INPAINT_NS)

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
