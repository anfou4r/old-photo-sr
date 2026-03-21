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
    """Remove scratches, cracks, fold marks, and damage from old photos.

    Uses multi-scale detection combining morphological operations, edge
    detection, and adaptive thresholding to find both fine scratches and
    heavy cracks. Runs multiple inpainting passes for thorough removal.

    Args:
        img: BGR uint8 numpy array.
        strength: Detection sensitivity (0-100). Higher = more aggressive.

    Returns:
        Cleaned BGR uint8 numpy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img

    # Blur to suppress texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # --- Multi-scale morphological detection ---
    # Use both white-hat (bright scratches) and black-hat (dark scratches)
    # with multiple kernel sizes and orientations for full coverage
    scratch_mask = np.zeros(gray.shape, dtype=np.float32)

    kernel_lengths = [11, 21, 31, 51]
    for length in kernel_lengths:
        for angle_kernel in [
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)),   # vertical
            cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)),   # horizontal
        ]:
            tophat_w = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, angle_kernel)
            tophat_b = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, angle_kernel)
            scratch_mask += tophat_w.astype(np.float32)
            scratch_mask += tophat_b.astype(np.float32)

        # Diagonal detection via rotated kernels (45° and 135°)
        diag_size = max(3, length // 3)
        for dx, dy in [(1, 1), (1, -1)]:
            kern = np.zeros((diag_size, diag_size), dtype=np.uint8)
            for k in range(diag_size):
                r = k if dy > 0 else diag_size - 1 - k
                kern[r, k] = 1
            tophat_w = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kern)
            tophat_b = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kern)
            scratch_mask += tophat_w.astype(np.float32)
            scratch_mask += tophat_b.astype(np.float32)

    # --- Edge-based detection for sharp crack boundaries ---
    # Lower Canny thresholds catch more crack edges
    canny_lo = max(20, 80 - int(strength * 0.6))
    canny_hi = canny_lo * 2
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    scratch_mask += edges.astype(np.float32) * 2.0

    # Normalize
    if scratch_mask.max() > 0:
        scratch_mask = scratch_mask / scratch_mask.max() * 255.0
    scratch_mask = scratch_mask.astype(np.uint8)

    # --- Adaptive thresholding ---
    # strength 0 -> thresh ~100 (conservative), strength 100 -> thresh ~20
    thresh = max(20, 100 - int(strength * 0.8))
    _, mask = cv2.threshold(scratch_mask, thresh, 255, cv2.THRESH_BINARY)

    # Dilate to cover the full width of cracks
    dilate_iters = max(1, 1 + strength // 30)  # 1-4 iterations
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, dilate_k, iterations=dilate_iters)

    # --- Component filtering (less strict than before) ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    max_area = img_area * 0.02  # Up to 2% of image per crack (was 0.1%)
    min_area = 10  # Filter tiny noise specks
    # Minimum aspect ratio depends on strength: lower = accept more shapes
    min_aspect = max(1.5, 5.0 - strength * 0.035)  # 5.0 -> 1.5

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        sw = stats[i, cv2.CC_STAT_WIDTH]
        sh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(sw, sh) / max(min(sw, sh), 1)

        if area < min_area:
            mask[labels == i] = 0
        elif area > max_area and aspect < min_aspect:
            # Only filter large blobs if they're not elongated
            mask[labels == i] = 0

    # --- Multi-pass inpainting for thorough crack removal ---
    inpaint_radius = max(3, min(10, 3 + strength // 15))
    result = img.copy()

    # First pass: main inpainting with Telea (better for large regions)
    result = cv2.inpaint(result, mask, inpaintRadius=inpaint_radius,
                         flags=cv2.INPAINT_TELEA)

    # Second pass with Navier-Stokes for smoothing residual artifacts
    # Re-detect remaining scratches on the inpainted result (catch leftovers)
    if strength >= 30:
        gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (5, 5), 1.0)
        residual_mask = np.zeros(gray2.shape, dtype=np.float32)
        for length in [21, 41]:
            for kern in [
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)),
            ]:
                residual_mask += cv2.morphologyEx(
                    blurred2, cv2.MORPH_TOPHAT, kern).astype(np.float32)
                residual_mask += cv2.morphologyEx(
                    blurred2, cv2.MORPH_BLACKHAT, kern).astype(np.float32)

        if residual_mask.max() > 0:
            residual_mask = residual_mask / residual_mask.max() * 255.0
        residual_mask = residual_mask.astype(np.uint8)
        _, mask2 = cv2.threshold(residual_mask, thresh + 10, 255,
                                 cv2.THRESH_BINARY)
        mask2 = cv2.dilate(mask2, dilate_k, iterations=1)
        # Only inpaint areas that overlap with original scratch regions
        mask2 = cv2.bitwise_and(mask2, cv2.dilate(mask, dilate_k, iterations=3))
        if mask2.any():
            result = cv2.inpaint(result, mask2, inpaintRadius=inpaint_radius,
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
