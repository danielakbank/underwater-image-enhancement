# src/dataset.py
#
# Responsible for everything related to loading and preparing image data.
# Think of this as the "data manager" of the project.

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

# All images will be resized to this shape before being used.
# 256x256 is a good balance — large enough to preserve detail,
# small enough to train on CPU/Colab without running out of memory.
IMAGE_SIZE = (256, 256)


# ── Core Functions ───────────────────────────────────────────────────────────

def get_image_pairs(raw_dir: str, ref_dir: str) -> list[tuple[Path, Path]]:
    """
    Scans both folders and returns a list of matched (raw, reference) path pairs.

    Matching is done by filename stem (the name without extension).
    For example: raw/1.png is matched with reference/1.png

    Args:
        raw_dir: Path to the folder containing degraded underwater images.
        ref_dir: Path to the folder containing clean reference images.

    Returns:
        A sorted list of (raw_path, reference_path) tuples.
    """
    raw_dir = Path(raw_dir)
    ref_dir = Path(ref_dir)

    # Build a dictionary of {filename_stem: full_path} for reference images
    # This makes matching O(1) — much faster than nested loops
    ref_map = {p.stem: p for p in ref_dir.iterdir() if p.is_file()}

    pairs = []
    for raw_path in sorted(raw_dir.iterdir()):
        if not raw_path.is_file():
            continue

        # Check if a matching reference image exists
        if raw_path.stem in ref_map:
            pairs.append((raw_path, ref_map[raw_path.stem]))
        else:
            print(f"  ⚠️  No reference found for: {raw_path.name} — skipping")

    print(f"  ✅  Found {len(pairs)} matched image pairs")
    return pairs


def load_image(path: str | Path, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Loads a single image from disk, resizes it, and normalises pixel values.

    OpenCV loads images in BGR format by default.
    We convert to RGB because that is what TensorFlow and Matplotlib expect.

    Pixel values are normalised from [0, 255] → [0.0, 1.0]
    Neural networks train significantly better on this range.

    Args:
        path:  Path to the image file.
        size:  Target (width, height) to resize to. Defaults to IMAGE_SIZE.

    Returns:
        A float32 NumPy array of shape (height, width, 3) with values in [0, 1].
    """
    img = cv2.imread(str(path))

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to target size
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Normalise to [0.0, 1.0] and cast to float32
    # float32 uses less memory than float64 and is what TensorFlow expects
    img = img.astype(np.float32) / 255.0

    return img


def load_pair(
    raw_path: str | Path,
    ref_path: str | Path,
    size: tuple = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a single (raw, reference) image pair.

    Args:
        raw_path: Path to the degraded image.
        ref_path: Path to the clean reference image.
        size:     Target resize dimensions.

    Returns:
        A tuple of (raw_image, reference_image) as float32 NumPy arrays.
    """
    raw = load_image(raw_path, size)
    ref = load_image(ref_path, size)
    return raw, ref


def load_dataset(
    raw_dir: str,
    ref_dir: str,
    size: tuple = IMAGE_SIZE,
    limit: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the full dataset into memory as two NumPy arrays.

    This is used during training — the CNN needs all images
    loaded into arrays of shape (N, H, W, C) where:
        N = number of images
        H = height (256)
        W = width  (256)
        C = channels (3 for RGB)

    Args:
        raw_dir: Path to degraded images folder.
        ref_dir: Path to reference images folder.
        size:    Target resize dimensions.
        limit:   Optional cap on number of pairs to load (useful for testing).

    Returns:
        (raw_images, ref_images) — two float32 arrays of shape (N, 256, 256, 3)
    """
    pairs = get_image_pairs(raw_dir, ref_dir)

    if limit:
        pairs = pairs[:limit]
        print(f"  ℹ️   Limiting to {limit} pairs for testing")

    raw_images = []
    ref_images = []

    for raw_path, ref_path in tqdm(pairs, desc="Loading images"):
        raw, ref = load_pair(raw_path, ref_path, size)
        raw_images.append(raw)
        ref_images.append(ref)

    # Stack list of arrays into a single 4D array
    # Shape becomes: (N, 256, 256, 3)
    return np.array(raw_images), np.array(ref_images)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test the dataset loader.
    Usage: python src/dataset.py
    """
    print("\n── Dataset Test ────────────────────────────────────────")

    RAW_DIR = "data/raw"
    REF_DIR = "data/reference"

    # Test 1 — pair matching
    pairs = get_image_pairs(RAW_DIR, REF_DIR)

    # Test 2 — load just 5 pairs to verify shapes and values
    raw_imgs, ref_imgs = load_dataset(RAW_DIR, REF_DIR, limit=5)

    print(f"\n  Raw images shape  : {raw_imgs.shape}")
    print(f"  Ref images shape  : {ref_imgs.shape}")
    print(f"  Pixel value range : {raw_imgs.min():.2f} → {raw_imgs.max():.2f}")
    print(f"  Data type         : {raw_imgs.dtype}")
    print("────────────────────────────────────────────────────────\n")