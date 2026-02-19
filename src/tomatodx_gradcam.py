"""
tomatodx_gradcam.py
====================
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities for TomatoDx.

Usage:
    python src/tomatodx_gradcam.py \
        --model_path ./outputs/tomatodx_best.keras \
        --image_path /path/to/leaf.jpg \
        --output_path ./outputs/gradcam_overlay.png

Reference:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. ICCV 2017.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Miner",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM Class
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given model and target convolutional layer.

    Grad-CAM works by:
    1. Forward-passing the image through the model.
    2. Computing the gradient of the predicted class score w.r.t. the
       feature maps of the target conv layer.
    3. Global-average-pooling the gradients to get per-channel weights.
    4. Computing a weighted sum of feature maps → raw heatmap.
    5. Applying ReLU (keep only positive activations) and normalizing.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str):
        """
        Args:
            model:      Trained Keras model.
            layer_name: Name of the target convolutional layer.
                        For EfficientNetB0 use 'top_conv' or 'block7a_project_conv'.
        """
        self.model = model
        self.layer_name = layer_name

        # Build a sub-model that outputs both the conv feature maps and final predictions
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

    def compute_heatmap(
        self, img_array: np.ndarray, class_idx: int = None
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image.

        Args:
            img_array:  Preprocessed image array of shape (1, H, W, 3).
            class_idx:  Target class index. If None, uses the predicted class.

        Returns:
            heatmap: Normalized heatmap of shape (H', W') in [0, 1].
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            # Watch the conv layer output
            conv_outputs, predictions = self.grad_model(img_tensor)
            tape.watch(conv_outputs)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            # Score for the target class
            class_score = predictions[:, class_idx]

        # Gradient of class score w.r.t. conv feature maps
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pooling of gradients → importance weights per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination of feature maps
        conv_outputs = conv_outputs[0]  # (H', W', C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H', W', 1)
        heatmap = tf.squeeze(heatmap)  # (H', W')

        # ReLU: keep only positive contributions
        heatmap = tf.nn.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap.numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess an image for model inference.

    Returns:
        img_array:    Preprocessed array (1, 224, 224, 3) in [0, 1].
        original_img: Original RGB image as numpy array for overlay.
    """
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array, original_img


def overlay_heatmap(
    heatmap: np.ndarray,
    original_img: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        heatmap:      Normalized heatmap array in [0, 1].
        original_img: Original RGB image as numpy array.
        alpha:        Transparency of the heatmap overlay (0=invisible, 1=opaque).
        colormap:     OpenCV colormap to use (default: JET).

    Returns:
        superimposed: RGB image with heatmap overlay.
    """
    # Resize heatmap to match original image dimensions
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed


def visualize_gradcam(
    model: tf.keras.Model,
    image_path: str,
    layer_name: str = "top_conv",
    output_path: str = None,
    show: bool = True,
) -> np.ndarray:
    """
    Full pipeline: load image → compute Grad-CAM → overlay → save/show.

    Args:
        model:       Trained Keras model.
        image_path:  Path to input image.
        layer_name:  Target conv layer name.
        output_path: If provided, saves the visualization to this path.
        show:        If True, displays the plot using matplotlib.

    Returns:
        overlay: Superimposed image as numpy array.
    """
    # Preprocess
    img_array, original_img = preprocess_image(image_path)

    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(preds[0])
    pred_class_name = CLASS_NAMES[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100

    print(f"\n{'='*50}")
    print(f"  Predicted Class : {pred_class_name}")
    print(f"  Confidence      : {confidence:.2f}%")
    print(f"  Target Layer    : {layer_name}")
    print(f"{'='*50}\n")

    # Compute Grad-CAM
    gradcam = GradCAM(model, layer_name)
    heatmap = gradcam.compute_heatmap(img_array, class_idx=pred_class_idx)

    # Overlay
    overlay = overlay_heatmap(heatmap, original_img, alpha=0.4)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM Visualization\nPredicted: {pred_class_name} ({confidence:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    # Heatmap with colorbar
    heatmap_display = axes[1].imshow(
        cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0])),
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(heatmap_display, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM visualization saved to: {output_path}")

    if show:
        plt.show()

    plt.close()
    return overlay


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Automatically find the name of the last convolutional layer in the model.
    Useful when the exact layer name is unknown.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,)):
            return layer.name
        # Handle EfficientNet's internal layers
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in model.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualization for a tomato leaf image."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved Keras model (.keras or .h5).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input leaf image.",
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="top_conv",
        help="Target convolutional layer name (default: 'top_conv' for EfficientNetB0).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/gradcam_overlay.png",
        help="Path to save the Grad-CAM overlay image.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot (useful for headless environments).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print("Model loaded successfully.")

    visualize_gradcam(
        model=model,
        image_path=args.image_path,
        layer_name=args.layer_name,
        output_path=args.output_path,
        show=not args.no_show,
    )
