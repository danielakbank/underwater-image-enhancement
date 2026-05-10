# app.py
#
# Gradio demo app for underwater image enhancement.
# Runs the full pipeline — classical CV and CNN — on any uploaded image.
# Generates a public shareable link via share=True.
#
# Usage:
#   python app.py
#
# Then open the local URL shown in terminal, or share the public link.

import os
import sys
import numpy as np
import gradio as gr
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for server use
import matplotlib.pyplot as plt

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from enhance import enhance_pipeline
from train import combined_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE  = (256, 256)
MODEL_PATH  = "models/unet_efficientnetb0.keras"
EXAMPLE_DIR = "assets"


# ── Load Model Once at Startup ────────────────────────────────────────────────
# We load the model once when the app starts — not on every request.
# This keeps inference fast (no repeated disk reads).

print("\n── Loading U-Net model ─────────────────────────────────")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"combined_loss": combined_loss}
    )
    print(f"  ✅ Model loaded — {model.count_params():,} parameters")
    CNN_AVAILABLE = True
except Exception as e:
    print(f"  ⚠️  Could not load model: {e}")
    print("     App will run in classical CV only mode")
    model = None
    CNN_AVAILABLE = False

print("────────────────────────────────────────────────────────\n")


# ── Image Processing Helpers ──────────────────────────────────────────────────

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Prepares an uploaded image for the pipeline.

    Gradio provides images as uint8 RGB arrays.
    We resize to 256x256 and normalise to [0, 1] float32.

    Args:
        image: uint8 RGB array from Gradio, any size.

    Returns:
        float32 RGB array, shape (256, 256, 3), values in [0, 1].
    """
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def postprocess(image: np.ndarray) -> np.ndarray:
    """
    Converts a float32 [0, 1] image back to uint8 [0, 255] for display.

    Args:
        image: float32 array, values in [0, 1].

    Returns:
        uint8 array, values in [0, 255].
    """
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def cnn_predict(image: np.ndarray) -> np.ndarray:
    """
    Runs a single image through the CNN model.

    Args:
        image: float32 RGB array, shape (256, 256, 3).

    Returns:
        Enhanced float32 RGB array, shape (256, 256, 3).
    """
    enhanced = model.predict(image[np.newaxis, ...], verbose=0)[0]
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


# ── Reference-Free Quality Metrics ───────────────────────────────────────────

def compute_perceptual_metrics(
    raw: np.ndarray,
    enhanced: np.ndarray
) -> dict:
    """
    Computes reference-free image quality indicators.

    Since we have no ground truth at inference time, we measure
    how much the enhancement improved perceptual quality compared
    to the raw input.

    Colourfulness: std of colour opponent channels — higher = more vibrant
    Contrast:      RMS contrast of luminance — higher = better contrast
    Sharpness:     Variance of Laplacian — higher = sharper edges

    Args:
        raw:      float32 RGB input image, values in [0, 1].
        enhanced: float32 RGB enhanced image, values in [0, 1].

    Returns:
        Dictionary with raw and enhanced scores for each metric.
    """
    def colourfulness(img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        return float(np.sqrt(np.std(rg)**2 + np.std(yb)**2))

    def contrast(img):
        gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        return float(np.std(gray))

    def sharpness(img):
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
        gray_uint8 = (gray * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        return float(np.var(laplacian))

    return {
        "colourfulness": {
            "raw"     : colourfulness(raw),
            "enhanced": colourfulness(enhanced),
        },
        "contrast": {
            "raw"     : contrast(raw),
            "enhanced": contrast(enhanced),
        },
        "sharpness": {
            "raw"     : sharpness(raw),
            "enhanced": sharpness(enhanced),
        },
    }


# ── Comparison Chart ──────────────────────────────────────────────────────────

def make_metrics_chart(
    raw: np.ndarray,
    classical: np.ndarray,
    cnn: np.ndarray,
) -> plt.Figure:
    """
    Creates a reference-free perceptual quality chart showing
    Raw vs Classical CV vs CNN across three quality indicators.

    All values shown as percentage of raw input so they are
    easy to interpret at a glance.

    Args:
        raw:       float32 raw input image.
        classical: float32 classical CV enhanced image.
        cnn:       float32 CNN enhanced image.

    Returns:
        Matplotlib Figure.
    """
    raw_m = compute_perceptual_metrics(raw, raw)
    cls_m = compute_perceptual_metrics(raw, classical)
    cnn_m = compute_perceptual_metrics(raw, cnn)

    metrics = ["colourfulness", "contrast", "sharpness"]
    labels  = ["Colourfulness", "Contrast", "Sharpness"]
    methods = ["Raw Input", "Classical CV", "CNN (U-Net)"]
    colours = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0f1117")

    for ax, metric, label in zip(axes, metrics, labels):
        values = [
            raw_m[metric]["raw"],
            cls_m[metric]["enhanced"],
            cnn_m[metric]["enhanced"],
        ]

        # Normalise to percentage of raw for easy reading
        baseline = values[0] if values[0] > 0 else 1.0
        pct = [v / baseline * 100 for v in values]

        bars = ax.bar(methods, pct, color=colours, width=0.5, zorder=3)
        ax.set_title(label, color="white", fontsize=12, fontweight="bold")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=9)
        ax.set_ylabel("% of Raw Input", color="white", fontsize=9)
        ax.axhline(
            y=100, color="#666", linestyle="--",
            linewidth=1, zorder=2, label="Raw baseline"
        )
        ax.set_ylim(0, max(pct) * 1.3)
        ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        for bar, val in zip(bars, pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold"
            )

    plt.suptitle(
        "Perceptual Quality Indicators (relative to raw input)",
        color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    return fig


# ── Main Enhancement Function ─────────────────────────────────────────────────

def enhance_image(image: np.ndarray):
    """
    Main function called by Gradio on every image upload.

    Takes a raw underwater image and returns:
        - Classical CV enhanced image
        - CNN enhanced image (if model loaded)
        - Perceptual quality chart
        - Text summary of scores and benchmark results

    Args:
        image: uint8 RGB array from Gradio file upload.

    Returns:
        Tuple of (classical_output, cnn_output, metrics_chart, summary_text)
    """
    if image is None:
        return None, None, None, "Please upload an image."

    # ── Preprocess ──
    img_float = preprocess(image)

    # ── Classical CV enhancement ──
    classical = enhance_pipeline(img_float)

    # ── CNN enhancement ──
    if CNN_AVAILABLE:
        cnn = cnn_predict(img_float)
    else:
        cnn = classical.copy()

    # ── Perceptual quality chart ──
    chart = make_metrics_chart(img_float, classical, cnn)

    # ── Per-method perceptual scores ──
    cls_m = compute_perceptual_metrics(img_float, classical)
    cnn_m = compute_perceptual_metrics(img_float, cnn)

    # ── Text summary ──
    summary = (
        "### Enhancement Results\n\n"
        "**Classical CV**\n"
        f"- Colourfulness : {cls_m['colourfulness']['enhanced']:.1f} "
        f"(raw: {cls_m['colourfulness']['raw']:.1f})\n"
        f"- Contrast      : {cls_m['contrast']['enhanced']:.3f} "
        f"(raw: {cls_m['contrast']['raw']:.3f})\n"
        f"- Sharpness     : {cls_m['sharpness']['enhanced']:.1f} "
        f"(raw: {cls_m['sharpness']['raw']:.1f})\n\n"
        "**CNN (U-Net)**\n"
        f"- Colourfulness : {cnn_m['colourfulness']['enhanced']:.1f} "
        f"(raw: {cnn_m['colourfulness']['raw']:.1f})\n"
        f"- Contrast      : {cnn_m['contrast']['enhanced']:.3f} "
        f"(raw: {cnn_m['contrast']['raw']:.3f})\n"
        f"- Sharpness     : {cnn_m['sharpness']['enhanced']:.1f} "
        f"(raw: {cnn_m['sharpness']['raw']:.1f})\n\n"
        "---\n"
        "📊 **UIEB Benchmark (50 images)**\n\n"
        "| Method | PSNR | SSIM |\n"
        "|---|---|---|\n"
        "| Raw Baseline | 15.73 dB | 0.7151 |\n"
        "| Classical CV | 17.13 dB | 0.8000 |\n"
        "| CNN (U-Net)  | 18.76 dB | 0.8141 |\n"
    )

    return (
        postprocess(classical),
        postprocess(cnn),
        chart,
        summary
    )


# ── Gradio Interface ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    """
    Builds the Gradio UI using Blocks for full layout control.

    Blocks gives us more control than gr.Interface —
    we can arrange components in rows and columns exactly
    how we want them.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(
        title="Underwater Image Enhancement",
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
        ),
        css="""
            .title { text-align: center; margin-bottom: 0.5rem; }
            .subtitle { text-align: center; color: #94a3b8;
                        margin-bottom: 1.5rem; font-size: 0.95rem; }
            .metric-box { background: #1e293b; border-radius: 8px;
                          padding: 1rem; margin-top: 0.5rem; }
        """
    ) as demo:

        # ── Header ──
        gr.Markdown(
            "# 🌊 Underwater Image Enhancement",
            elem_classes="title"
        )
        gr.Markdown(
            "Upload a degraded underwater image to enhance it using "
            "classical computer vision and a trained U-Net CNN "
            "(EfficientNetB0 encoder, pretrained on ImageNet).",
            elem_classes="subtitle"
        )

        # ── Main Layout ──
        with gr.Row():

            # Left column — input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📁 Upload Underwater Image",
                    type="numpy",
                    height=300,
                )
                enhance_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg"
                )

                # Example images from assets/ folder
                if os.path.exists(EXAMPLE_DIR):
                    examples = [
                        os.path.join(EXAMPLE_DIR, f)
                        for f in os.listdir(EXAMPLE_DIR)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        and "sample" not in f.lower()
                        and "history" not in f.lower()
                        and "metrics" not in f.lower()
                    ]
                    if examples:
                        gr.Examples(
                            examples=examples[:3],
                            inputs=input_image,
                            label="Example Images"
                        )

            # Right column — outputs
            with gr.Column(scale=2):
                with gr.Row():
                    classical_output = gr.Image(
                        label="🔵 Classical CV Enhanced",
                        height=280,
                    )
                    cnn_output = gr.Image(
                        label="🟢 CNN Enhanced (U-Net)",
                        height=280,
                    )

        # ── Metrics Row ──
        with gr.Row():
            with gr.Column(scale=2):
                metrics_chart = gr.Plot(
                    label="📊 Perceptual Quality Indicators"
                )
            with gr.Column(scale=1):
                summary_text = gr.Markdown(
                    elem_classes="metric-box"
                )

        # ── Info Accordion ──
        with gr.Accordion("ℹ️ About this project", open=False):
            gr.Markdown("""
            ## Underwater Image Enhancement Pipeline

            **Dataset:** UIEB (Underwater Image Enhancement Benchmark) — 890 paired images

            **Pipeline:**
            1. **Classical CV** — Grey World colour correction → CLAHE contrast enhancement → Unsharp masking
            2. **CNN** — U-Net with pretrained EfficientNetB0 encoder (transfer learning, ImageNet weights)

            **Evaluation on UIEB benchmark (50 images):**
            | Method | PSNR (dB) | SSIM |
            |---|---|---|
            | Raw Baseline | 15.73 | 0.7151 |
            | Classical CV | 17.13 | 0.8000 |
            | CNN (U-Net)  | 18.76 | 0.8141 |

            **Training:** Two-phase transfer learning on Google Colab T4 GPU
            - Phase 1: Decoder-only training (20 epochs, encoder frozen)
            - Phase 2: Full fine-tuning (20 epochs, encoder unfrozen, LR/10)

            **Built with:** Python · TensorFlow · OpenCV · NumPy · Gradio
            """)

        # ── Button Action ──
        enhance_btn.click(
            fn=enhance_image,
            inputs=[input_image],
            outputs=[classical_output, cnn_output, metrics_chart, summary_text],
        )

        # Also trigger on image upload for instant feedback
        input_image.change(
            fn=enhance_image,
            inputs=[input_image],
            outputs=[classical_output, cnn_output, metrics_chart, summary_text],
        )

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()