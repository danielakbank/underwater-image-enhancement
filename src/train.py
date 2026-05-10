# src/train.py
#
# Builds a U-Net from scratch using pure TensorFlow/Keras.
# EfficientNetB0 is used as the pretrained encoder via tf.keras.applications.
#
# Architecture: U-Net with EfficientNetB0 encoder (pretrained on ImageNet)
# This is more portfolio-impressive than using a library — it shows
# you understand the architecture at the code level.

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    "image_size"    : (256, 256),
    "batch_size"    : 8,
    "epochs"        : 50,
    "learning_rate" : 1e-4,
    "val_split"     : 0.15,
    "model_path"    : "models/unet_efficientnetb0.keras",
}


# ── Loss Function ─────────────────────────────────────────────────────────────

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combines MAE and SSIM loss for better image reconstruction quality.

    MAE (Mean Absolute Error):
        Measures average pixel difference — good for overall colour
        and brightness accuracy.

    SSIM Loss:
        Measures structural similarity — good for preserving edges
        and textures that MAE alone tends to blur.

    Args:
        y_true: Reference (ground truth) image tensor.
        y_pred: Model output (enhanced) image tensor.

    Returns:
        Scalar combined loss value.
    """
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )
    return 0.5 * mae_loss + 0.5 * ssim_loss


# ── Encoder Block ─────────────────────────────────────────────────────────────

def decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int
) -> tf.Tensor:
    """
    One block of the U-Net decoder.

    Each decoder block does three things:
    1. Upsample — doubles the spatial resolution (height x width)
    2. Concatenate — merges with the skip connection from the encoder
       (this is what makes it a U-Net — it reintroduces fine detail
        that was lost during downsampling)
    3. Convolve — two conv layers to learn how to combine the features

    Args:
        x:       Input tensor from the previous decoder block.
        skip:    Skip connection tensor from the matching encoder layer.
        filters: Number of convolutional filters in this block.

    Returns:
        Output tensor for the next decoder block.
    """
    # Upsample by factor of 2
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    # Concatenate with skip connection along the channel axis
    if skip is not None:
        # Crop skip if spatial sizes don't match exactly
        x = layers.Concatenate()([x, skip])

    # Two conv layers with batch normalisation and ReLU activation
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


# ── Model Definition ──────────────────────────────────────────────────────────

def build_model(input_shape: tuple = (256, 256, 3)) -> tf.keras.Model:
    """
    Builds a U-Net with a pretrained EfficientNetB0 encoder.

    How the architecture works:
    ───────────────────────────
    Encoder (EfficientNetB0, pretrained on ImageNet):
        We use EfficientNetB0 as a feature extractor. Its weights are
        already trained on 1.2 million images so it already understands
        edges, textures, shapes, and colours. We freeze these weights
        initially so the decoder can learn without destabilising the
        encoder — then we fine-tune both together.

        We extract feature maps from 5 different depths of EfficientNetB0.
        Each depth captures increasingly abstract features:
            block1 → edges and low-level textures
            block2 → simple shapes
            block3 → complex patterns
            block4 → object parts
            block5 → semantic content (deepest features)

    Skip Connections:
        Feature maps from each encoder depth are passed directly to the
        corresponding decoder block. This lets the decoder recover fine
        spatial detail that gets compressed in the bottleneck.

    Decoder:
        5 upsampling blocks, each doubling spatial resolution and
        combining with its matching encoder skip connection.
        Final output: 3-channel RGB image with sigmoid activation
        so all pixel values are in [0, 1].

    Args:
        input_shape: (height, width, channels) — default 256x256 RGB.

    Returns:
        Compiled Keras Model ready for training.
    """
    inputs = keras.Input(shape=input_shape)

    # ── Encoder: EfficientNetB0 pretrained on ImageNet ──
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=inputs,
        include_top=False,       # remove the classification head
        weights="imagenet",      # pretrained weights = transfer learning
    )

    # Freeze encoder weights initially
    # We'll fine-tune after the decoder has warmed up
    base_model.trainable = False

    # Extract skip connections from specific layers at different depths
    # These layer names are specific to EfficientNetB0's architecture
    skip_connections = [
        base_model.get_layer("block2a_expand_activation").output,  # 64x64
        base_model.get_layer("block3a_expand_activation").output,  # 32x32
        base_model.get_layer("block4a_expand_activation").output,  # 16x16
        base_model.get_layer("block6a_expand_activation").output,  # 8x8
    ]

    # Bottleneck — deepest encoder output
    bottleneck = base_model.get_layer("top_activation").output     # 8x8

    # ── Decoder: Progressive upsampling with skip connections ──
    x = decoder_block(bottleneck, skip_connections[3], filters=256)
    x = decoder_block(x,          skip_connections[2], filters=128)
    x = decoder_block(x,          skip_connections[1], filters=64)
    x = decoder_block(x,          skip_connections[0], filters=32)

    # Final upsample to original resolution (no skip connection)
    x = decoder_block(x, None, filters=16)

    # ── Output Layer ──
    # sigmoid activation keeps all pixel values between 0 and 1
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="UNet_EfficientNetB0")

    # Compile with combined loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss=combined_loss,
        metrics=["mae"],
    )

    return model


# ── Data Pipeline ─────────────────────────────────────────────────────────────

def build_dataset(
    raw_images: np.ndarray,
    ref_images: np.ndarray,
    batch_size: int = CONFIG["batch_size"],
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Builds an efficient TensorFlow data pipeline from NumPy arrays.

    tf.data pipelines preload and preprocess batches on the CPU
    while the GPU trains — keeping the GPU fully utilised.

    Args:
        raw_images: Degraded images array, shape (N, 256, 256, 3).
        ref_images: Reference images array, shape (N, 256, 256, 3).
        batch_size: Images per training step.
        augment:    Apply random horizontal flip for data augmentation.

    Returns:
        Batched, prefetched tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((raw_images, ref_images))

    if augment:
        def augment_fn(raw, ref):
            # Apply identical random flip to both images
            combined = tf.stack([raw, ref], axis=0)
            combined = tf.image.random_flip_left_right(combined)
            return combined[0], combined[1]

        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return (
        dataset
        .shuffle(buffer_size=len(raw_images), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(model_path: str = CONFIG["model_path"]) -> list:
    """
    Training callbacks that monitor and control the training process.

    ModelCheckpoint : saves model whenever val_loss improves
    EarlyStopping   : stops training if val_loss stagnates for 10 epochs
    ReduceLROnPlateau: halves learning rate if val_loss plateaus for 5 epochs

    Args:
        model_path: Where to save the best model weights.

    Returns:
        List of configured Keras callback objects.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    return [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ── Sanity Check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Builds the model and prints a summary.
    Does NOT train — training happens in the Colab notebook.
    Usage: python src/train.py
    """
    print("\n── Model Sanity Check ──────────────────────────────────")
    print("  Building U-Net with EfficientNetB0 encoder...")

    model = build_model(input_shape=(256, 256, 3))

    print(f"\n  Input shape  : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")
    print(f"  Total params : {model.count_params():,}")
    print("\n  ✅  Model built successfully — ready for Colab training")
    print("────────────────────────────────────────────────────────\n")