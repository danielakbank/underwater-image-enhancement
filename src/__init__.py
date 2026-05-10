# src/__init__.py
# Run this file directly to verify your environment is set up correctly.
# Usage: python src/__init__.py

def check_environment():
    packages = {
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "tensorflow": "TensorFlow",
        "skimage": "scikit-image",
        "matplotlib": "Matplotlib",
        "gradio": "Gradio",
        "tqdm": "tqdm",
    }

    print("\n── Environment Check ───────────────────")
    all_good = True

    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✅  {name}")
        except ImportError:
            print(f"  ❌  {name} — run: pip install -r requirements.txt")
            all_good = False

    print("────────────────────────────────────────")
    print("  ✅  All good — ready to build!\n" if all_good else "  ❌  Fix missing packages above.\n")


if __name__ == "__main__":
    check_environment()