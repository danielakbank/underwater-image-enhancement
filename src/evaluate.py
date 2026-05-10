# src/evaluate.py
#
# Evaluates enhancement performance across three stages:
#   1. Raw baseline    — degraded image vs reference
#   2. Classical CV    — our preprocessing pipeline vs reference
#   3. CNN (U-Net)     — trained model output vs reference
#
# This three-way comparison is the core scientific contribution
# of the project and the centrepiece of the portfolio.

import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ── Individual Metric Functions ───────────────────────────────────────────────

def compute_psnr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Computes PSNR between reference and enhanced image.

    Higher is better. Measured in decibels (dB).
    Typical good values for underwater enhancement: 18–25 dB.

    Args:
        reference: Clean reference image, float32, values in [0, 1].
        enhanced:  Enhanced output image, float32, values in [0, 1].

    Returns:
        PSNR value in decibels.
    """
    return psnr(reference, enhanced, data_range=1.0)


def compute_ssim(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Computes SSIM between reference and enhanced image.

    Range: 0.0 → 1.0. Higher is better.
    Good enhancement results typically score above 0.85.

    Args:
        reference: Clean reference image, float32, values in [0, 1].
        enhanced:  Enhanced output image, float32, values in [0, 1].

    Returns:
        SSIM score between 0.0 and 1.0.
    """
    return ssim(reference, enhanced, channel_axis=2, data_range=1.0)


def compute_metrics(
    reference: np.ndarray,
    enhanced: np.ndarray
) -> dict[str, float]:
    """
    Computes both PSNR and SSIM and returns them as a dictionary.

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


# ── CNN Inference ─────────────────────────────────────────────────────────────

def load_cnn_model(model_path: str = "models/unet_efficientnetb0.keras"):
    """
    Loads the trained U-Net model from disk.

    We load it once here and pass it to evaluate_all() to avoid
    reloading it for every image — that would be very slow.

    Args:
        model_path: Path to the saved .keras model file.

    Returns:
        Loaded Keras model ready for inference.
    """
    import tensorflow as tf

    # Register combined_loss so Keras can deserialise the model
    # It needs to know about custom functions used during training
    from train import combined_loss

    print(f"  Loading CNN model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"combined_loss": combined_loss}
    )
    print(f"  ✅ Model loaded — {model.count_params():,} parameters")
    return model


def predict_single(model, image: np.ndarray) -> np.ndarray:
    """
    Runs a single image through the CNN and returns the enhanced output.

    Adds and removes the batch dimension that the model requires.
    Input shape:  (256, 256, 3)
    Model input:  (1, 256, 256, 3)  ← batch dimension added
    Model output: (1, 256, 256, 3)
    Return shape: (256, 256, 3)     ← batch dimension removed

    Args:
        model: Loaded Keras U-Net model.
        image: Single float32 image, shape (256, 256, 3), values in [0, 1].

    Returns:
        Enhanced float32 image, shape (256, 256, 3), values in [0, 1].
    """
    enhanced = model.predict(image[np.newaxis, ...], verbose=0)[0]
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


# ── Three-Way Evaluation ──────────────────────────────────────────────────────

def evaluate_all(
    raw_images: np.ndarray,
    ref_images: np.ndarray,
    enhance_fn: callable,
    model=None,
) -> dict[str, dict]:
    """
    Runs three-way evaluation across all image pairs.

    For each pair computes PSNR and SSIM for:
        - Raw baseline  (no enhancement)
        - Classical CV  (enhance_fn applied)
        - CNN           (model applied, if provided)

    Args:
        raw_images:  Array of degraded images, shape (N, 256, 256, 3).
        ref_images:  Array of reference images, shape (N, 256, 256, 3).
        enhance_fn:  Classical CV enhancement function.
        model:       Loaded CNN model. If None, CNN evaluation is skipped.

    Returns:
        Dictionary with keys 'baseline', 'classical', 'cnn'.
        Each value is a dict with 'psnr' and 'ssim' lists.
    """
    results = {
        "baseline" : {"psnr": [], "ssim": []},
        "classical": {"psnr": [], "ssim": []},
        "cnn"      : {"psnr": [], "ssim": []},
    }

    for raw, ref in tqdm(
        zip(raw_images, ref_images),
        total=len(raw_images),
        desc="  Evaluating"
    ):
        # ── Baseline: raw vs reference ──
        m = compute_metrics(ref, raw)
        results["baseline"]["psnr"].append(m["psnr"])
        results["baseline"]["ssim"].append(m["ssim"])

        # ── Classical CV ──
        classical = enhance_fn(raw)
        m = compute_metrics(ref, classical)
        results["classical"]["psnr"].append(m["psnr"])
        results["classical"]["ssim"].append(m["ssim"])

        # ── CNN ──
        if model is not None:
            cnn_enhanced = predict_single(model, raw)
            m = compute_metrics(ref, cnn_enhanced)
            results["cnn"]["psnr"].append(m["psnr"])
            results["cnn"]["ssim"].append(m["ssim"])

    return results


def summarise(results: dict) -> dict[str, dict]:
    """
    Converts per-image score lists into mean summary statistics.

    Args:
        results: Raw results dict from evaluate_all().

    Returns:
        Summary dict with mean PSNR and SSIM per method.
    """
    summary = {}
    for method, scores in results.items():
        if not scores["psnr"]:
            continue
        summary[method] = {
            "psnr": float(np.mean(scores["psnr"])),
            "ssim": float(np.mean(scores["ssim"])),
        }
    return summary


def print_results(summary: dict) -> None:
    """
    Prints the three-way comparison table in a clean format.

    Args:
        summary: Summary dict from summarise().
    """
    method_labels = {
        "baseline" : "Raw Baseline",
        "classical": "Classical CV",
        "cnn"      : "CNN (U-Net) ",
    }

    baseline_psnr = summary.get("baseline", {}).get("psnr", 0)
    baseline_ssim = summary.get("baseline", {}).get("ssim", 0)

    print("\n── Three-Way Evaluation Results ────────────────────────")
    print(f"  {'Method':<18} {'PSNR (dB)':>10} {'SSIM':>8} {'ΔPSNR':>8} {'ΔSSIM':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for method, label in method_labels.items():
        if method not in summary:
            continue
        p = summary[method]["psnr"]
        s = summary[method]["ssim"]
        dp = p - baseline_psnr
        ds = s - baseline_ssim

        # Don't show delta for baseline itself
        delta_p = f"{dp:+.2f}" if method != "baseline" else "  —"
        delta_s = f"{ds:+.4f}" if method != "baseline" else "  —"

        print(f"  {label:<18} {p:>10.2f} {s:>8.4f} {delta_p:>8} {delta_s:>8}")

    print("────────────────────────────────────────────────────────\n")


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Runs three-way evaluation on 50 image pairs.
    Usage: python src/evaluate.py
    """
    import sys
    import os
    sys.path.insert(0, "src")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    from dataset import load_dataset
    from enhance import enhance_pipeline

    print("\n── Three-Way Evaluation ────────────────────────────────")

    # ── Load data ──
    raw_imgs, ref_imgs = load_dataset(
        raw_dir="data/raw",
        ref_dir="data/reference",
        limit=50       # 50 images gives reliable averages without taking too long
    )

    # ── Load CNN model ──
    model = load_cnn_model("models/unet_efficientnetb0.keras")

    # ── Run evaluation ──
    results = evaluate_all(
        raw_images=raw_imgs,
        ref_images=ref_imgs,
        enhance_fn=enhance_pipeline,
        model=model,
    )

    # ── Summarise and print ──
    summary = summarise(results)
    print_results(summary)