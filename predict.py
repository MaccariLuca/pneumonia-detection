"""
Inference script.
Runs prediction on a single image or every image in a folder.

Usage:
    python predict.py --image path/to/image.jpeg
    python predict.py --folder path/to/folder
"""

import argparse
import sys
import os
import glob

import numpy as np
import cv2
import tf_keras as keras

from config import IMG_SIZE, MODEL_NAME


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_model() -> keras.Model:
    if not os.path.exists(MODEL_NAME):
        sys.exit(f"❌ Model not found: {MODEL_NAME}\n   Train the model first with: python train.py")
    print(f"✅ Loaded model: {MODEL_NAME}")
    return keras.models.load_model(MODEL_NAME)


def preprocess(image_path: str) -> np.ndarray | None:
    """Reads, resizes, normalizes a single image. Returns None on failure."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"   ⚠️  Could not read: {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # (1, H, W, 1)


def predict_single(model: keras.Model, image_path: str) -> None:
    """Prints prediction for one image."""
    x = preprocess(image_path)
    if x is None:
        return

    probs = model.predict(x, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(probs)]

    print(f"\n{'─' * 40}")
    print(f"  Image:      {os.path.basename(image_path)}")
    print(f"  Prediction: {predicted_class}")
    print(f"  Normal:     {probs[0]*100:.2f}%")
    print(f"  Pneumonia:  {probs[1]*100:.2f}%")
    print(f"{'─' * 40}")


def predict_folder(model: keras.Model, folder: str) -> None:
    """Runs prediction on every image in a folder and prints a summary."""
    extensions = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    if not files:
        sys.exit(f"❌ No images found in: {folder}")

    print(f"\n📂 Processing {len(files)} images in {folder}...\n")

    results = {"NORMAL": 0, "PNEUMONIA": 0}

    for path in sorted(files):
        x = preprocess(path)
        if x is None:
            continue

        probs = model.predict(x, verbose=0)[0]
        label = CLASS_NAMES[np.argmax(probs)]
        results[label] += 1
        print(f"   {label:>10}  ({probs[0]*100:5.1f}% / {probs[1]*100:5.1f}%)  — {os.path.basename(path)}")

    total = sum(results.values())
    print(f"\n📊 Summary:")
    print(f"   NORMAL:    {results['NORMAL']:4d}  ({results['NORMAL']/total*100:.1f}%)")
    print(f"   PNEUMONIA: {results['PNEUMONIA']:4d}  ({results['PNEUMONIA']/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Pneumonia detection inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        predict_single(model, args.image)
    else:
        predict_folder(model, args.folder)


if __name__ == "__main__":
    main()
