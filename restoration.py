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
    # Move towards neutral (128) with 50% strength (gentle correction)
    a_ch = np.clip(a_ch - (a_mean - 128) * 0.5, 0, 255)
    b_ch = np.clip(b_ch - (b_mean - 128) * 0.5, 0, 255)

    # CLAHE on L channel for gentle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_uint8 = np.clip(l_ch, 0, 255).astype(np.uint8)
    l_eq = clahe.apply(l_uint8).astype(np.float32)

    lab_corrected = cv2.merge([l_eq, a_ch.astype(np.uint8).astype(np.float32),
                                b_ch.astype(np.uint8).astype(np.float32)])
    corrected = cv2.cvtColor(lab_corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return corrected


def remove_scratches(img, strength=50):
    """Remove scratches, fold marks, and spots from old photos.

    Uses edge-based detection focused on thin linear artifacts,
    with strict filtering to avoid damaging photo content.

    Args:
        img: BGR uint8 numpy array.
        strength: Detection sensitivity (0-100). Higher = more aggressive.

    Returns:
        Cleaned BGR uint8 numpy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]

    # Blur to reduce texture noise before detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect thin linear structures using morphological top-hat
    # Top-hat highlights thin bright/dark lines against background
    scratch_mask = np.zeros(gray.shape, dtype=np.float32)

    for length in [21, 31]:
        # Vertical scratches (thin horizontal kernel for top-hat)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        tophat_v = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_v)
        scratch_mask += tophat_v.astype(np.float32)

        # Horizontal scratches
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        tophat_h = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_h)
        scratch_mask += tophat_h.astype(np.float32)

    # Normalize to [0, 255]
    if scratch_mask.max() > 0:
        scratch_mask = (scratch_mask / scratch_mask.max() * 255.0)
    scratch_mask = scratch_mask.astype(np.uint8)

    # High threshold to only catch obvious scratches
    # strength 0 -> thresh 120, strength 100 -> thresh 40
    thresh = max(40, 120 - int(strength * 0.8))
    _, mask = cv2.threshold(scratch_mask, thresh, 255, cv2.THRESH_BINARY)

    # Thin the mask to single-pixel width for clean detection
    mask = cv2.ximgproc.thinning(mask) if hasattr(cv2, 'ximgproc') else mask

    # Dilate slightly so inpainting can cover the scratch width
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, dilate_k, iterations=1)

    # Strict filtering: only keep very elongated, thin components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    max_area = h_img * w_img * 0.001  # Max 0.1% of image per scratch
    min_area = 20  # Ignore tiny specks
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        sw = stats[i, cv2.CC_STAT_WIDTH]
        sh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(sw, sh) / max(min(sw, sh), 1)
        # Must be elongated (aspect >= 5), not too large, not too small
        if area > max_area or area < min_area or aspect < 5:
            mask[labels == i] = 0

    # Small inpaint radius to minimize blurring
    inpaint_radius = max(2, min(5, 2 + strength // 30))
    result = cv2.inpaint(img, mask, inpaintRadius=inpaint_radius,
                         flags=cv2.INPAINT_NS)

    # Blend with original to reduce artifacts (keep 70-90% of inpainted)
    blend = 0.7 + (strength / 100.0) * 0.2  # 0.7 to 0.9
    result = cv2.addWeighted(result, blend, img, 1.0 - blend, 0)

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
