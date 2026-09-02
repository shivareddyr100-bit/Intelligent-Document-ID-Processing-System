"""
Train a CNN document classifier on your own data.

Usage:
    python train_classifier.py --data_dir ./training_data --epochs 20

Expected folder structure:
    training_data/
        Aadhaar/    (*.jpg, *.png)
        PAN/        (*.jpg, *.png)
        Unknown/    (*.jpg, *.png)   ← optional
"""

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Train document classifier CNN")
    p.add_argument("--data_dir", type=str, required=True, help="Root folder of image classes")
    p.add_argument("--output", type=str, default="models/weights/document_classifier.keras")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    return p.parse_args()


def build_model(num_classes: int = 3, input_size: tuple = (224, 224, 3)):
    """Build a small CNN with MobileNetV2 backbone."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError("TensorFlow is required for training.  pip install tensorflow")

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_size, include_top=False, weights="imagenet",
    )
    base.trainable = False  # Freeze backbone

    inputs = tf.keras.Input(shape=input_size)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset(data_dir: str, img_size: int):
    """Load images from class sub-folders using OpenCV (no TF dependency for I/O)."""
    import cv2

    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images, labels = [], []
    for cls_name in classes:
        folder = data_dir / cls_name
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(class_to_idx[cls_name])

    return np.array(images), np.array(labels), classes


def main():
    args = parse_args()

    print(f"Loading data from {args.data_dir} …")
    X, y, class_names = load_dataset(args.data_dir, args.img_size)
    print(f"  Found {len(X)} images across {len(class_names)} classes: {class_names}")

    if len(X) == 0:
        print("ERROR: No images found. Check --data_dir structure.")
        return

    model = build_model(num_classes=len(class_names))
    model.summary()

    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(str(out_path), save_best_only=True),
    ]

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    print(f"\n✅ Model saved to {out_path}")


if __name__ == "__main__":
    main()