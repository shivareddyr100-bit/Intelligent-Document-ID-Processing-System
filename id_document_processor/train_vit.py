"""
Train a Vision Transformer (ViT) for document classification.

Usage:
    python train_vit.py --data_dir ./training_data --epochs 10

Expected structure:
    training_data/
        Aadhaar/*.jpg
        PAN/*.jpg
        Unknown/*.jpg   (optional)
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)


class DocDataset(Dataset):
    """Simple image folder dataset for HuggingFace Trainer."""

    def __init__(self, root_dir: str, processor):
        self.processor = processor
        self.samples = []
        self.label_map = {}

        root = Path(root_dir)
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.label_map = {name: idx for idx, name in enumerate(class_names)}
        self.class_names = class_names

        for cls_name in class_names:
            folder = root / cls_name
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), self.label_map[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["labels"] = torch.tensor(label)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label),
        }


def parse_args():
    p = argparse.ArgumentParser(description="Train ViT document classifier")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="models/weights/vit_classifier")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.model_name}")
    processor = ViTImageProcessor.from_pretrained(args.model_name)

    # Build dataset
    dataset = DocDataset(args.data_dir, processor)
    num_labels = len(dataset.class_names)
    print(f"Classes ({num_labels}): {dataset.class_names}")
    print(f"Samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No images found.")
        return

    # Load model with new classification head
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: n for i, n in enumerate(dataset.class_names)},
        label2id={n: i for i, n in enumerate(dataset.class_names)},
        ignore_mismatched_sizes=True,
    )

    # Training
    out_path = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nTraining ViT...")
    trainer.train()

    # Save
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    print(f"\n✅ ViT model saved to {out_path}")


if __name__ == "__main__":
    main()