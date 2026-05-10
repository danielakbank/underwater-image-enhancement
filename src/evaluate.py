# src/evaluate.py
#
# Computes image quality metrics to measure enhancement performance.
# We use two complementary metrics: PSNR and SSIM.
#
# These metrics compare the enhanced image against the clean reference,
# giving us objective, quantifiable proof of improvement.

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from tqdm import tqdm


# ── Individual Metric Functions ───────────────────────────────────────────────

def compute_psnr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Computes PSNR (Peak Signal-to-Noise Ratio) between two images.

    PSNR measures the ratio between the maximum possible pixel value
    and the mean squared error between the two images.

    Higher is better. Measured in decibels (dB).
    - Below 20 dB : poor quality
    - 20–30 dB    : acceptable
    - 30–40 dB    : good
    - Above 40 dB : excellent

    Args:
        reference: Clean reference image, float32, values in [0, 1].
        enhanced:  Enhanced output image, float32, values in [0, 1].

    Returns:
        PSNR value in decibels (float).
    """
    # data_range is the difference between max and min possible values
    # Since our images are normalised to [0, 1], data_range = 1.0
    return psnr(reference, enhanced, data_range=1.0)


def compute_ssim(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Computes SSIM (Structural Similarity Index) between two images.

    SSIM evaluates three components locally across the image:
        - Luminance : how similar the brightness is
        - Contrast  : how similar the contrast is
        - Structure : how similar the patterns/textures are

    Range: 0.0 (completely different) → 1.0 (identical)
    Good enhancement results typically score above 0.85.

    Args:
        reference: Clean reference image, float32, values in [0, 1].
        enhanced:  Enhanced output image, float32, values in [0, 1].

    Returns:
        SSIM value between 0.0 and 1.0 (float).
    """
    # channel_axis=2 tells skimage this is an RGB image (channels are axis 2)
    # data_range=1.0 because our images are normalised to [0, 1]
    return ssim(reference, enhanced, channel_axis=2, data_range=1.0)


def compute_metrics(
    reference: np.ndarray,
    enhanced: np.ndarray
) -> dict[str, float]:
    """
    Computes both PSNR and SSIM and returns them as a dictionary.

    This is the main function called from other modules.

    Args:
        reference: Clean reference image, float32, values in [0, 1].
        enhanced:  Enhanced output image, float32, values in [0, 1].

    Returns:
        Dictionary with keys 'psnr' and 'ssim'.
    """
    return {
        "psnr": compute_psnr(reference, enhanced),
        "ssim": compute_ssim(reference, enhanced),
    }


# ── Batch Evaluation ──────────────────────────────────────────────────────────

def evaluate_dataset(
    raw_images: np.ndarray,
    ref_images: np.ndarray,
    enhance_fn: callable,
) -> dict[str, float]:
    """
    Evaluates enhancement performance across a batch of image pairs.

    Applies enhance_fn to each raw image, then computes PSNR and SSIM
    against the reference. Returns the average scores across all pairs.

    We compute two comparisons:
        - Baseline : raw image vs reference (before enhancement)
        - Enhanced : enhanced image vs reference (after enhancement)

    This lets us prove that our pipeline actually improves things.

    Args:
        raw_images:  Array of degraded images, shape (N, H, W, 3).
        ref_images:  Array of reference images, shape (N, H, W, 3).
        enhance_fn:  Function that takes a raw image and returns enhanced image.

    Returns:
        Dictionary with baseline and enhanced average PSNR and SSIM scores.
    """
    baseline_psnr_scores = []
    baseline_ssim_scores = []
    enhanced_psnr_scores = []
    enhanced_ssim_scores = []

    for raw, ref in tqdm(
        zip(raw_images, ref_images),
        total=len(raw_images),
        desc="Evaluating"
    ):
        # Baseline — how bad is the raw image compared to reference?
        baseline_metrics = compute_metrics(ref, raw)
        baseline_psnr_scores.append(baseline_metrics["psnr"])
        baseline_ssim_scores.append(baseline_metrics["ssim"])

        # Enhanced — how much does our pipeline improve things?
        enhanced = enhance_fn(raw)
        enhanced_metrics = compute_metrics(ref, enhanced)
        enhanced_psnr_scores.append(enhanced_metrics["psnr"])
        enhanced_ssim_scores.append(enhanced_metrics["ssim"])

    return {
        "baseline_psnr": float(np.mean(baseline_psnr_scores)),
        "baseline_ssim": float(np.mean(baseline_ssim_scores)),
        "enhanced_psnr": float(np.mean(enhanced_psnr_scores)),
        "enhanced_ssim": float(np.mean(enhanced_ssim_scores)),
        "psnr_improvement": float(
            np.mean(enhanced_psnr_scores) - np.mean(baseline_psnr_scores)
        ),
        "ssim_improvement": float(
            np.mean(enhanced_ssim_scores) - np.mean(baseline_ssim_scores)
        ),
    }


def print_results(results: dict) -> None:
    """
    Prints evaluation results in a clean, readable format.

    Args:
        results: Dictionary returned by evaluate_dataset().
    """
    print("\n── Evaluation Results ──────────────────────────────────")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Enhanced':>10} {'Improvement':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
    print(
        f"  {'PSNR (dB)':<20} "
        f"{results['baseline_psnr']:>10.2f} "
        f"{results['enhanced_psnr']:>10.2f} "
        f"{results['psnr_improvement']:>+12.2f}"
    )
    print(
        f"  {'SSIM':<20} "
        f"{results['baseline_ssim']:>10.4f} "
        f"{results['enhanced_ssim']:>10.4f} "
        f"{results['ssim_improvement']:>+12.4f}"
    )
    print("────────────────────────────────────────────────────────\n")


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to evaluate the classical pipeline on 20 images.
    Usage: python src/evaluate.py
    """
    import sys
    sys.path.insert(0, "src")

    from dataset import load_dataset
    from enhance import enhance_pipeline

    print("\n── Running Evaluation on 20 image pairs ───────────────")

    raw_imgs, ref_imgs = load_dataset(
        raw_dir="data/raw",
        ref_dir="data/reference",
        limit=20
    )

    results = evaluate_dataset(raw_imgs, ref_imgs, enhance_pipeline)
    print_results(results)