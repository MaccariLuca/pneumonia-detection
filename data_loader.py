"""
Data loading and preprocessing module.
Handles image loading, normalization, and train/val splitting.
"""

import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

from config import IMG_SIZE, DATASET_PATH, TEST_PATH


def load_images_from_folder(folder: str, label: int, max_images: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads grayscale images from a folder and resizes them.

    Args:
        folder:     Path to the image folder.
        label:      Class label (0 = NORMAL, 1 = PNEUMONIA).
        max_images: Optional cap on number of images to load.

    Returns:
        images: numpy array of shape (N, IMG_SIZE, IMG_SIZE), dtype uint8.
        labels: numpy array of shape (N,), dtype int32.
    """
    extensions = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(folder, ext)))

    print(f"   Found {len(file_list)} images in {os.path.basename(folder)}")

    if max_images:
        file_list = file_list[:max_images]

    images, labels = [], []

    for i, path in enumerate(file_list):
        if i % 500 == 0 and i > 0:
            print(f"      Loading: {i}/{len(file_list)}")
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
        except Exception:
            continue

    print(f"   ✅ Loaded {len(images)} images")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def load_dataset(dataset_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads full train and validation sets, normalizes to [0, 1], and shuffles training data.

    Returns:
        X_train, y_train, X_val, y_val
    """
    # --- Training ---
    print("\n[1/6] Loading training dataset...")
    normal_imgs, normal_labels = load_images_from_folder(os.path.join(dataset_path, "NORMAL"), 0)
    pneumonia_imgs, pneumonia_labels = load_images_from_folder(os.path.join(dataset_path, "PNEUMONIA"), 1)

    if len(normal_imgs) == 0 or len(pneumonia_imgs) == 0:
        raise ValueError("Training dataset is empty. Check DATASET_PATH in config.py")

    X_train = np.concatenate([normal_imgs, pneumonia_imgs]).astype(np.float32) / 255.0
    y_train = np.concatenate([normal_labels, pneumonia_labels]).astype(np.int32)
    X_train = np.expand_dims(X_train, axis=-1)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print(f"   Training: {len(normal_imgs)} NORMAL + {len(pneumonia_imgs)} PNEUMONIA")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")

    # --- Validation ---
    print("\n[2/6] Loading validation dataset...")
    normal_val, normal_val_labels = load_images_from_folder(os.path.join(test_path, "NORMAL"), 0)
    pneumonia_val, pneumonia_val_labels = load_images_from_folder(os.path.join(test_path, "PNEUMONIA"), 1)

    if len(normal_val) == 0 or len(pneumonia_val) == 0:
        raise ValueError("Validation dataset is empty. Check TEST_PATH in config.py")

    X_val = np.concatenate([normal_val, pneumonia_val]).astype(np.float32) / 255.0
    y_val = np.concatenate([normal_val_labels, pneumonia_val_labels]).astype(np.int32)
    X_val = np.expand_dims(X_val, axis=-1)

    print(f"   Validation: {len(normal_val)} NORMAL + {len(pneumonia_val)} PNEUMONIA")
    print(f"   X_val: {X_val.shape} | y_val: {y_val.shape}")

    return X_train, y_train, X_val, y_val


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Computes balanced class weights manually.
    Formula: n_samples / (n_classes * count_per_class), with sqrt to moderate.

    Args:
        y: Label array.

    Returns:
        Dictionary mapping class index to weight.
    """
    unique, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique)

    weights = {}
    for cls, cnt in zip(unique, counts):
        weights[int(cls)] = float((n_samples / (n_classes * cnt)) ** 0.5)

    print(f"\n[3/6] Class distribution:")
    for cls, cnt in zip(unique, counts):
        name = "NORMAL" if cls == 0 else "PNEUMONIA"
        print(f"   {name}: {cnt} ({cnt / n_samples * 100:.1f}%)")
    print(f"   Weights: {weights}")

    return weights
