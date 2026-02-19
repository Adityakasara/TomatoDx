"""
tomatodx_train.py
==================
Standalone training script for TomatoDx: Tomato Leaf Disease Identification.

This script mirrors the training logic in TomatoDx_Project.ipynb and can be
run from the command line for reproducible, headless training.

Usage:
    python src/tomatodx_train.py \
        --data_dir ./data/tomato \
        --epochs 40 \
        --batch_size 32 \
        --output_dir ./outputs \
        [--fine_tune_epochs 20] \
        [--seed 42]

Dataset structure expected:
    data_dir/
    ├── train/
    │   ├── Tomato_Bacterial_spot/
    │   ├── Tomato_Early_blight/
    │   └── ...
    ├── val/
    └── test/
"""

import argparse
import os
import random
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] All random seeds set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
NUM_CLASSES = 11

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
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def build_data_generators(
    data_dir: str, batch_size: int
) -> tuple:
    """
    Build Keras ImageDataGenerators for train, validation, and test sets.

    Data Augmentation (train only — no augmentation on val/test to avoid leakage):
        - Horizontal & vertical flips
        - Random rotation (±20°)
        - Width/height shift (±10%)
        - Zoom (±15%)
        - Shear (±10%)
        - Brightness adjustment (±20%)
        - Fill mode: 'nearest'

    Normalization: pixel values scaled to [0, 1] for all splits.

    Args:
        data_dir:   Root directory containing train/, val/, test/ subdirs.
        batch_size: Number of samples per batch.

    Returns:
        train_gen, val_gen, test_gen: Keras DirectoryIterators.
    """
    # ── Training augmentation ────────────────────────────────────────────────
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    # ── Validation / Test: only rescale, NO augmentation ────────────────────
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"\n[Data] Train samples  : {train_gen.samples}")
    print(f"[Data] Val samples    : {val_gen.samples}")
    print(f"[Data] Test samples   : {test_gen.samples}")
    print(f"[Data] Classes found  : {list(train_gen.class_indices.keys())}\n")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build the TomatoDx model using EfficientNetB0 as the backbone.

    Architecture Design Rationale:
    ─────────────────────────────
    • EfficientNetB0: Chosen for its compound scaling (width, depth, resolution),
      achieving state-of-the-art accuracy with ~5.3M parameters — ideal for
      deployment on resource-constrained devices.
    • ImageNet pretraining: Provides rich low-level feature representations
      (edges, textures) that transfer well to plant disease patterns.
    • GlobalAveragePooling2D: Reduces spatial dimensions while retaining
      channel-wise information; more robust than Flatten for varying input sizes.
    • BatchNormalization: Stabilizes training and reduces internal covariate shift.
    • Dropout(0.3 / 0.2): Regularization to prevent overfitting on the
      relatively small PlantVillage dataset.
    • Dense(256, ReLU): Task-specific feature transformation layer.
    • Dense(num_classes, Softmax): Multi-class probability output.

    Args:
        num_classes: Number of output classes (default: 11).

    Returns:
        model: Compiled Keras model with base layers frozen.
    """
    # ── Base Model (frozen) ──────────────────────────────────────────────────
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # Freeze all base layers in Phase 1

    # ── Classification Head ──────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)  # training=False keeps BN in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="TomatoDx_EfficientNetB0")

    # ── Compile ──────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[Model] Total parameters       : {model.count_params():,}")
    print(f"[Model] Trainable parameters   : {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    print(f"[Model] Non-trainable params   : {sum(tf.size(v).numpy() for v in model.non_trainable_variables):,}\n")

    return model, base_model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def build_callbacks(output_dir: str, phase: int = 1) -> list:
    """
    Build training callbacks.

    Callbacks:
        • ModelCheckpoint: Save best model by val_accuracy.
        • EarlyStopping: Stop if val_accuracy doesn't improve for `patience` epochs.
        • ReduceLROnPlateau: Halve LR if val_loss plateaus for 3 epochs.
        • TensorBoard: Log metrics for visualization.

    Args:
        output_dir: Directory to save model checkpoints and logs.
        phase:      Training phase (1=head training, 2=fine-tuning).

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "tomatodx_best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5 if phase == 1 else 7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "logs", f"phase{phase}"),
            histogram_freq=0,
        ),
    ]

    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_phase1(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    epochs: int,
    output_dir: str,
) -> tf.keras.callbacks.History:
    """
    Phase 1: Train only the classification head with the base frozen.

    Rationale: Training the head first allows it to learn task-specific
    features before the base weights are modified, preventing catastrophic
    forgetting of ImageNet representations.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Training Classification Head (Base Frozen)")
    print("=" * 60)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=build_callbacks(output_dir, phase=1),
        verbose=1,
    )

    return history


def train_phase2(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_gen,
    val_gen,
    epochs: int,
    output_dir: str,
    fine_tune_layers: int = 30,
) -> tf.keras.callbacks.History:
    """
    Phase 2: Unfreeze top layers of the base model for fine-tuning.

    Rationale: Fine-tuning the top convolutional layers allows the model to
    adapt high-level features (shapes, textures) to the tomato disease domain.
    Lower layers (edges, gradients) are kept frozen as they are domain-agnostic.

    Args:
        fine_tune_layers: Number of top layers to unfreeze (default: 30).
    """
    print("\n" + "=" * 60)
    print(f"  PHASE 2: Fine-Tuning Top {fine_tune_layers} Layers of EfficientNetB0")
    print("=" * 60)

    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Recompile with a much lower learning rate to avoid destroying learned weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"[Fine-tune] Trainable parameters: {trainable_count:,}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=build_callbacks(output_dir, phase=2),
        verbose=1,
    )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: tf.keras.Model,
    test_gen,
    output_dir: str,
    class_names: list,
) -> dict:
    """
    Evaluate the model on the test set and save metrics.

    Metrics computed:
        • Overall test accuracy
        • Per-class Precision, Recall, F1-score (macro average)
        • Confusion matrix (saved as heatmap PNG)
        • Full classification report (saved as CSV)

    Args:
        model:       Trained Keras model.
        test_gen:    Test data generator.
        output_dir:  Directory to save evaluation outputs.
        class_names: List of class name strings.

    Returns:
        metrics: Dict with accuracy, precision, recall, f1.
    """
    print("\n[Evaluation] Running inference on test set...")
    test_gen.reset()

    # Get predictions
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # ── Overall Accuracy ─────────────────────────────────────────────────────
    accuracy = np.mean(y_pred == y_true)

    # ── Precision, Recall, F1 (macro) ────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n{'─'*50}")
    print(f"  Test Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Macro Precision: {precision * 100:.2f}%")
    print(f"  Macro Recall   : {recall * 100:.2f}%")
    print(f"  Macro F1-Score : {f1 * 100:.2f}%")
    print(f"{'─'*50}\n")

    # ── Full Classification Report ────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"[Evaluation] Classification report saved to: {report_path}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, output_dir: str) -> None:
    """Plot and save the confusion matrix as a heatmap."""
    # Normalize by row (true labels) for better readability
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    short_names = [name.replace("Tomato_", "").replace("_", "\n") for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_normalized],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=short_names,
            yticklabels=short_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(axis="both", labelsize=8)

    plt.suptitle("TomatoDx — Test Set Evaluation", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(
    history1: tf.keras.callbacks.History,
    history2: tf.keras.callbacks.History,
    output_dir: str,
) -> None:
    """
    Plot and save training vs validation accuracy and loss curves.

    Both phases are concatenated for a continuous view. A vertical dashed line
    marks the transition from Phase 1 (head training) to Phase 2 (fine-tuning).

    Analysis:
        • If val_accuracy closely tracks train_accuracy → good generalization.
        • Large gap (train >> val) → overfitting; consider more dropout/augmentation.
        • Both curves plateau early → underfitting; consider more capacity/epochs.
    """
    # Concatenate histories
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
    phase1_end = len(history1.history["accuracy"])

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("TomatoDx — Training History", fontsize=14, fontweight="bold")

    # ── Accuracy ─────────────────────────────────────────────────────────────
    axes[0].plot(epochs_range, acc, "b-o", markersize=3, label="Train Accuracy")
    axes[0].plot(epochs_range, val_acc, "r-o", markersize=3, label="Val Accuracy")
    axes[0].axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7, label="Phase 2 Start")
    axes[0].set_title("Accuracy vs Epoch", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ── Loss ─────────────────────────────────────────────────────────────────
    axes[1].plot(epochs_range, loss, "b-o", markersize=3, label="Train Loss")
    axes[1].plot(epochs_range, val_loss, "r-o", markersize=3, label="Val Loss")
    axes[1].axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7, label="Phase 2 Start")
    axes[1].set_title("Loss vs Epoch", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Training history saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="TomatoDx: Train EfficientNetB0 on tomato leaf disease dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory with train/, val/, test/ subdirectories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for Phase 1 (head training). Default: 20.",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=20,
        help="Number of epochs for Phase 2 (fine-tuning). Default: 20.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training. Default: 32.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model, plots, and reports. Default: ./outputs.",
    )
    parser.add_argument(
        "--fine_tune_layers",
        type=int,
        default=30,
        help="Number of top EfficientNetB0 layers to unfreeze in Phase 2. Default: 30.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  TomatoDx — Training Script")
    print(f"{'='*60}")
    print(f"  Data dir         : {args.data_dir}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Phase 1 epochs   : {args.epochs}")
    print(f"  Phase 2 epochs   : {args.fine_tune_epochs}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Fine-tune layers : {args.fine_tune_layers}")
    print(f"  Seed             : {args.seed}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = build_data_generators(args.data_dir, args.batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    model, base_model = build_model(num_classes=NUM_CLASSES)
    model.summary()

    # ── Phase 1: Head Training ───────────────────────────────────────────────
    history1 = train_phase1(model, train_gen, val_gen, args.epochs, args.output_dir)

    # ── Phase 2: Fine-Tuning ─────────────────────────────────────────────────
    history2 = train_phase2(
        model, base_model, train_gen, val_gen,
        args.fine_tune_epochs, args.output_dir, args.fine_tune_layers
    )

    # ── Load Best Model ──────────────────────────────────────────────────────
    best_model_path = os.path.join(args.output_dir, "tomatodx_best.keras")
    print(f"\n[Model] Loading best model from: {best_model_path}")
    best_model = tf.keras.models.load_model(best_model_path)

    # ── Evaluation ───────────────────────────────────────────────────────────
    metrics = evaluate_model(best_model, test_gen, args.output_dir, CLASS_NAMES)

    # ── Training Curves ──────────────────────────────────────────────────────
    plot_training_history(history1, history2, args.output_dir)

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"  Best model saved to : {best_model_path}")
    print(f"  Test Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score      : {metrics['f1']*100:.2f}%")
    print(f"{'='*60}\n")
