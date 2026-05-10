# src/enhance.py
#
# Classical computer vision enhancement pipeline.
# Each function tackles one specific problem with underwater images.
# The full pipeline applies all three in sequence.

import cv2
import numpy as np


# ── Step 1: Colour Correction ─────────────────────────────────────────────────

def correct_colour(image: np.ndarray) -> np.ndarray:
    """
    Corrects the blue/green colour cast common in underwater images.

    Water absorbs red light first, then green, then blue.
    This means underwater images are dominated by blue/green tones
    with very little red. We fix this using the Grey World Assumption:
    "In a natural image, the average colour of the scene should be grey."

    We scale each channel so its mean matches the overall image mean.
    This effectively balances the colours back to neutral.

    Args:
        image: Float32 RGB image with pixel values in [0, 1].

    Returns:
        Colour-corrected float32 RGB image with values clipped to [0, 1].
    """
    # Calculate the mean of the entire image (across all channels)
    overall_mean = np.mean(image)

    # Calculate the mean of each individual channel
    # axis=(0,1) means we average across height and width, keeping channels
    channel_means = np.mean(image, axis=(0, 1))

    # Scale each channel so its mean matches the overall mean
    # Channels with low means (like red) get boosted
    # Channels with high means (like blue) get reduced
    scale = overall_mean / (channel_means + 1e-6)  # 1e-6 prevents division by zero
    corrected = image * scale

    # Clip to valid range — scaling can push values above 1.0 or below 0.0
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


# ── Step 2: Contrast Enhancement ─────────────────────────────────────────────

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhances local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalisation).

    Why CLAHE instead of simple histogram equalisation?
    Simple equalisation works globally — it can over-brighten already
    bright areas and create unnatural looking images.

    CLAHE works locally — it divides the image into small tiles (8x8)
    and equalises each tile independently, then blends the tiles together.
    The "contrast limited" part prevents noise from being over-amplified.

    CLAHE works on single-channel images, so we:
    1. Convert RGB → LAB colour space
    2. Apply CLAHE only to the L (lightness) channel
    3. Convert back to RGB

    This way we enhance brightness/contrast without distorting colours.

    Args:
        image: Float32 RGB image with pixel values in [0, 1].

    Returns:
        Contrast-enhanced float32 RGB image with values in [0, 1].
    """
    # Convert from float32 [0,1] to uint8 [0,255] for OpenCV
    img_uint8 = (image * 255).astype(np.uint8)

    # Convert RGB → LAB colour space
    # LAB separates lightness (L) from colour (A=green-red, B=blue-yellow)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

    # Split into individual channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    # clipLimit: controls contrast amplification limit (2.0 is a safe value)
    # tileGridSize: size of the local tiles (8x8 pixels)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only to the lightness channel
    l_enhanced = clahe.apply(l_channel)

    # Merge channels back together
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # Convert LAB → RGB
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Convert back to float32 [0, 1]
    return (enhanced / 255.0).astype(np.float32)


# ── Step 3: Deblurring ────────────────────────────────────────────────────────

def deblur(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Sharpens the image using Unsharp Masking.

    How it works:
    1. Create a blurred copy of the image (Gaussian blur)
    2. Subtract the blurred copy from the original
       → This isolates the fine detail / edge information
    3. Add that detail back to the original, amplified by 'strength'

    Formula: sharpened = original + strength × (original - blurred)

    The result is an image with more defined edges and texture.

    Args:
        image:    Float32 RGB image with pixel values in [0, 1].
        strength: How much sharpening to apply. 0.0 = none, 1.0 = strong.
                  Default 0.5 gives natural-looking results for underwater.

    Returns:
        Sharpened float32 RGB image with values clipped to [0, 1].
    """
    # Apply Gaussian blur — kernel size (5,5) and sigma=1.0 are standard values
    # The kernel size must be odd numbers
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

    # Calculate the detail layer (what the blur removed)
    detail = image - blurred

    # Add the detail back, amplified by strength
    sharpened = image + strength * detail

    # Clip to valid range
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def enhance_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Applies the full classical enhancement pipeline in sequence:
        1. Colour correction
        2. Contrast enhancement
        3. Deblurring

    This is the main function you will call from other modules
    and from the Gradio app.

    Args:
        image: Float32 RGB image with pixel values in [0, 1].

    Returns:
        Fully enhanced float32 RGB image with values in [0, 1].
    """
    image = correct_colour(image)
    image = enhance_contrast(image)
    image = deblur(image)
    return image


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test the enhancement pipeline.
    Processes 3 images and saves results to data/results/
    Usage: python src/enhance.py
    """
    import matplotlib.pyplot as plt
    from dataset import get_image_pairs, load_image
    from pathlib import Path

    print("\n── Enhancement Pipeline Test ───────────────────────────")

    RAW_DIR = "data/raw"
    REF_DIR = "data/reference"
    OUT_DIR = Path("data/results")
    OUT_DIR.mkdir(exist_ok=True)

    pairs = get_image_pairs(RAW_DIR, REF_DIR)[:3]

    for i, (raw_path, ref_path) in enumerate(pairs):
        raw = load_image(raw_path)
        enhanced = enhance_pipeline(raw)

        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(raw)
        axes[0].set_title("Raw (Degraded)")
        axes[0].axis("off")

        axes[1].imshow(enhanced)
        axes[1].set_title("Enhanced (Classical CV)")
        axes[1].axis("off")

        plt.suptitle(f"Sample {i + 1}: {raw_path.name}", fontsize=12)
        plt.tight_layout()

        out_path = OUT_DIR / f"classical_sample_{i + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✅  Saved: {out_path}")

    print("────────────────────────────────────────────────────────\n")