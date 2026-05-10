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
    Uses Grey World Assumption with a strength limiter to prevent
    overcorrection on images that don't need heavy adjustment.

    Args:
        image: Float32 RGB image with pixel values in [0, 1].

    Returns:
        Colour-corrected float32 RGB image with values clipped to [0, 1].
    """
    overall_mean = np.mean(image)
    channel_means = np.mean(image, axis=(0, 1))
    scale = overall_mean / (channel_means + 1e-6)

    # Limit how aggressively we scale any single channel.
    # Without this cap, the red channel gets boosted too strongly
    # on images that are already reasonably balanced.
    scale = np.clip(scale, 0.8, 1.3)

    corrected = image * scale
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


# ── Step 2: Contrast Enhancement ─────────────────────────────────────────────

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhances local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalisation).

    CLAHE divides the image into small tiles and equalises contrast
    locally in each tile, then blends them together. This avoids
    over-brightening already bright areas — a common problem with
    global histogram equalisation.

    We apply CLAHE only to the L (lightness) channel of the LAB
    colour space so we enhance brightness without distorting colours.

    Args:
        image: Float32 RGB image with pixel values in [0, 1].

    Returns:
        Contrast-enhanced float32 RGB image with values in [0, 1].
    """
    # Convert float32 [0, 1] → uint8 [0, 255] for OpenCV
    img_uint8 = (image * 255).astype(np.uint8)

    # Convert RGB → LAB colour space
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

    # Split into individual channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    # clipLimit=2.0 — safe value that avoids noise amplification
    # tileGridSize=(8,8) — size of local tiles in pixels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only to the lightness channel
    l_enhanced = clahe.apply(l_channel)

    # Merge channels back and convert LAB → RGB
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Convert back to float32 [0, 1]
    return (enhanced / 255.0).astype(np.float32)


# ── Step 3: Deblurring ────────────────────────────────────────────────────────

def deblur(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Sharpens the image using Unsharp Masking.

    How it works:
    1. Create a blurred copy of the image
    2. Subtract the blurred copy from the original — this isolates edges
    3. Add that edge detail back to the original, amplified by strength

    Formula: sharpened = original + strength × (original - blurred)

    Args:
        image:    Float32 RGB image with pixel values in [0, 1].
        strength: How much sharpening to apply. Default 0.5 is natural
                  looking for underwater images without amplifying noise.

    Returns:
        Sharpened float32 RGB image with values clipped to [0, 1].
    """
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    detail = image - blurred
    sharpened = image + strength * detail
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def enhance_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Applies the full classical enhancement pipeline in sequence:
        1. Colour correction  — fixes blue/green cast
        2. Contrast enhancement — improves local contrast with CLAHE
        3. Deblurring         — sharpens edges with unsharp masking

    This is the main function called from other modules and the Gradio app.

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