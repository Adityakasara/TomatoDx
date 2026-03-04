"""
tomatodx_inference.py
======================
Load a saved TomatoDx model and run inference on a single leaf image.

Usage:
    python src/tomatodx_inference.py \
        --model_path ./outputs/tomatodx_best.keras \
        --image_path /path/to/leaf.jpg \
        [--top_k 3]
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
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

# Disease descriptions for user-friendly output
DISEASE_INFO = {
    "Tomato_Bacterial_spot": "Caused by Xanthomonas spp. Appears as water-soaked lesions that turn brown.",
    "Tomato_Early_blight": "Caused by Alternaria solani. Concentric ring lesions on older leaves.",
    "Tomato_Late_blight": "Caused by Phytophthora infestans. Dark, water-soaked lesions with white mold.",
    "Tomato_Leaf_Miner": "Caused by Liriomyza spp. Serpentine mines visible on leaf surface.",
    "Tomato_Leaf_Mold": "Caused by Passalora fulva. Pale green/yellow spots on upper leaf surface.",
    "Tomato_Septoria_leaf_spot": "Caused by Septoria lycopersici. Small circular spots with dark borders.",
    "Tomato_Spider_mites": "Caused by Tetranychus urticae. Stippling and bronzing of leaves.",
    "Tomato_Target_Spot": "Caused by Corynespora cassiicola. Concentric ring lesions resembling a target.",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "Viral disease transmitted by whiteflies. Causes leaf curling and yellowing.",
    "Tomato_Tomato_mosaic_virus": "Viral disease causing mosaic pattern of light and dark green on leaves.",
    "Tomato_healthy": "No disease detected. The leaf appears healthy.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for model inference.

    Steps:
        1. Open image and convert to RGB (handles RGBA, grayscale, etc.)
        2. Resize to 224×224 (EfficientNetB0 input size)
        3. Normalize pixel values to [0, 1]
        4. Add batch dimension → shape (1, 224, 224, 3)

    Args:
        image_path: Path to the input image file.

    Returns:
        img_array: Preprocessed numpy array of shape (1, 224, 224, 3).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    model: tf.keras.Model, img_array: np.ndarray, top_k: int = 3
) -> list[dict]:
    """
    Run model inference and return top-k predictions.

    Args:
        model:     Loaded Keras model.
        img_array: Preprocessed image array (1, 224, 224, 3).
        top_k:     Number of top predictions to return.

    Returns:
        predictions: List of dicts with keys 'class', 'confidence', 'info'.
    """
    preds = model.predict(img_array, verbose=0)[0]  # Shape: (num_classes,)

    # Get top-k indices sorted by confidence (descending)
    top_k_indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for class_index in top_k_indices:
        class_name = CLASS_NAMES[class_index]
        results.append(
            {
                "class": class_name,
                "confidence": float(preds[class_index]) * 100,
                "info": DISEASE_INFO.get(class_name, "No information available."),
            }
        )

    return results


def print_results(predictions: list[dict], image_path: str) -> None:
    """Pretty-print inference results to the console."""
    print("\n" + "=" * 60)
    print("  TomatoDx — Disease Identification Results")
    print("=" * 60)
    print(f"  Image: {os.path.basename(image_path)}")
    print("-" * 60)

    for rank, pred in enumerate(predictions, start=1):
        bar_len = int(pred["confidence"] / 5)  # Scale to 20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n  Rank #{rank}")
        print(f"  Class      : {pred['class']}")
        print(f"  Confidence : {pred['confidence']:.2f}%  [{bar}]")
        print(f"  Info       : {pred['info']}")

    print("\n" + "=" * 60)

    # Final verdict
    top_pred = predictions[0]
    if top_pred["confidence"] >= 70:
        verdict = "HIGH CONFIDENCE"
    elif top_pred["confidence"] >= 40:
        verdict = "MODERATE CONFIDENCE — consider expert review"
    else:
        verdict = "LOW CONFIDENCE — image quality may be poor"

    print(f"\n  ✦ Verdict: {top_pred['class']}")
    print(f"    {verdict} ({top_pred['confidence']:.1f}%)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="TomatoDx: Run inference on a tomato leaf image."
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
        help="Path to the input tomato leaf image.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load Model ──────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print(f"Model loaded. Input shape: {model.input_shape}")

    # ── Preprocess Image ────────────────────────────────────────────────────
    print(f"Processing image: {args.image_path}")
    img_array = preprocess_image(args.image_path)

    # ── Predict ─────────────────────────────────────────────────────────────
    predictions = predict(model, img_array, top_k=args.top_k)

    # ── Display Results ─────────────────────────────────────────────────────
    print_results(predictions, args.image_path)
