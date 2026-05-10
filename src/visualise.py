# src/visualise.py
#
# Generates visual comparisons and metric charts.
# Produces publication-quality figures for the portfolio and Gradio app.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── Styling ───────────────────────────────────────────────────────────────────

# Apply a clean style globally for all plots
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})


# ── Single Image Comparison ───────────────────────────────────────────────────

def plot_comparison(
    raw: np.ndarray,
    enhanced: np.ndarray,
    reference: np.ndarray,
    metrics: dict = None,
    title: str = "Enhancement Comparison",
    save_path: str = None,
) -> plt.Figure:
    """
    Plots a three-panel side-by-side comparison:
        Panel 1: Raw degraded image
        Panel 2: Enhanced output
        Panel 3: Clean reference image

    Optionally overlays PSNR and SSIM scores on the enhanced panel.
    Optionally saves the figure to disk.

    Args:
        raw:       Degraded input image, float32 [0, 1].
        enhanced:  Enhanced output image, float32 [0, 1].
        reference: Clean reference image, float32 [0, 1].
        metrics:   Optional dict with keys 'psnr' and 'ssim'.
        title:     Figure title string.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [raw, enhanced, reference]
    labels = ["Raw (Degraded)", "Enhanced (Pipeline)", "Reference (Ground Truth)"]
    border_colours = ["#e74c3c", "#2ecc71", "#3498db"]

    for ax, img, label, colour in zip(axes, images, labels, border_colours):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

        # Add a coloured border to visually separate panels
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)

    # Overlay metrics on the enhanced panel if provided
    if metrics:
        axes[1].set_xlabel(
            f"PSNR: {metrics['psnr']:.2f} dB   |   SSIM: {metrics['ssim']:.4f}",
            fontsize=11,
            color="#2ecc71",
            fontweight="bold",
            labelpad=8,
        )

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅  Saved: {save_path}")

    return fig


# ── Metrics Bar Chart ─────────────────────────────────────────────────────────

def plot_metrics(
    results: dict,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots a dual bar chart comparing baseline vs enhanced PSNR and SSIM.

    Having a chart like this in your portfolio gives reviewers an
    immediate visual grasp of how much your pipeline improves things.

    Args:
        results:   Dictionary returned by evaluate.evaluate_dataset().
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    categories = ["Baseline (Raw)", "Enhanced"]
    colours = ["#e74c3c", "#2ecc71"]

    # ── PSNR Bar Chart ──
    psnr_values = [results["baseline_psnr"], results["enhanced_psnr"]]
    bars = axes[0].bar(categories, psnr_values, color=colours, width=0.5, zorder=3)
    axes[0].set_title("PSNR Comparison", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)", fontsize=11)
    axes[0].set_ylim(0, max(psnr_values) * 1.2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Add value labels on top of each bar
    for bar, val in zip(bars, psnr_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f} dB",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    # Add improvement arrow annotation
    improvement = results["psnr_improvement"]
    axes[0].annotate(
        f"{improvement:+.2f} dB",
        xy=(1, psnr_values[1]),
        xytext=(0.5, (psnr_values[0] + psnr_values[1]) / 2),
        fontsize=10, color="#27ae60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        ha="center"
    )

    # ── SSIM Bar Chart ──
    ssim_values = [results["baseline_ssim"], results["enhanced_ssim"]]
    bars = axes[1].bar(categories, ssim_values, color=colours, width=0.5, zorder=3)
    axes[1].set_title("SSIM Comparison", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("SSIM Score", fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, ssim_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    improvement = results["ssim_improvement"]
    axes[1].annotate(
        f"{improvement:+.4f}",
        xy=(1, ssim_values[1]),
        xytext=(0.5, (ssim_values[0] + ssim_values[1]) / 2),
        fontsize=10, color="#27ae60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        ha="center"
    )

    plt.suptitle(
        "Classical CV Enhancement — Metric Improvements",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅  Saved: {save_path}")

    return fig


# ── Multi-Sample Grid ─────────────────────────────────────────────────────────

def plot_sample_grid(
    raw_images: np.ndarray,
    enhanced_images: np.ndarray,
    reference_images: np.ndarray,
    n_samples: int = 4,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots a grid of N samples, each showing raw / enhanced / reference.

    This is the most visually impactful figure for your portfolio —
    showing consistent improvement across multiple diverse images.

    Args:
        raw_images:       Array of degraded images, shape (N, H, W, 3).
        enhanced_images:  Array of enhanced images, shape (N, H, W, 3).
        reference_images: Array of reference images, shape (N, H, W, 3).
        n_samples:        Number of rows to show (default 4).
        save_path:        If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    n_samples = min(n_samples, len(raw_images))
    fig = plt.figure(figsize=(15, 4 * n_samples))

    # GridSpec gives us precise control over subplot layout
    gs = gridspec.GridSpec(
        n_samples, 3,
        figure=fig,
        hspace=0.05,
        wspace=0.05
    )

    col_titles = ["Raw (Degraded)", "Enhanced", "Reference"]
    col_colours = ["#e74c3c", "#2ecc71", "#3498db"]

    for row in range(n_samples):
        images = [raw_images[row], enhanced_images[row], reference_images[row]]

        for col, (img, title, colour) in enumerate(
            zip(images, col_titles, col_colours)
        ):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")

            # Column headers on first row only
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", color=colour, pad=8)

            # Row label on left column only
            if col == 0:
                ax.set_ylabel(
                    f"Sample {row + 1}",
                    fontsize=10,
                    rotation=90,
                    labelpad=8
                )

    plt.suptitle(
        "Underwater Enhancement — Multi-Sample Results",
        fontsize=15, fontweight="bold", y=1.01
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅  Saved: {save_path}")

    return fig


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to generate all visualisation outputs.
    Usage: python src/visualise.py
    """
    import sys
    sys.path.insert(0, "src")

    from dataset import load_dataset
    from enhance import enhance_pipeline
    from evaluate import evaluate_dataset, compute_metrics

    print("\n── Generating Visualisations ───────────────────────────")

    # Load 8 image pairs
    raw_imgs, ref_imgs = load_dataset(
        raw_dir="data/raw",
        ref_dir="data/reference",
        limit=8
    )

    # Enhance all 8
    enhanced_imgs = np.array([enhance_pipeline(img) for img in raw_imgs])

    # ── Figure 1: Single comparison with metrics ──
    metrics = compute_metrics(ref_imgs[0], enhanced_imgs[0])
    plot_comparison(
        raw=raw_imgs[0],
        enhanced=enhanced_imgs[0],
        reference=ref_imgs[0],
        metrics=metrics,
        title="Underwater Enhancement — Single Sample",
        save_path="data/results/comparison_single.png"
    )

    # ── Figure 2: Metrics bar chart ──
    results = evaluate_dataset(raw_imgs, ref_imgs, enhance_pipeline)
    plot_metrics(
        results=results,
        save_path="data/results/metrics_chart.png"
    )

    # ── Figure 3: Multi-sample grid ──
    plot_sample_grid(
        raw_images=raw_imgs,
        enhanced_images=enhanced_imgs,
        reference_images=ref_imgs,
        n_samples=4,
        save_path="data/results/sample_grid.png"
    )

    print("\n  All figures saved to data/results/")
    print("────────────────────────────────────────────────────────\n")