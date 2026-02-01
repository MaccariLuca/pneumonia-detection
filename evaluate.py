"""
Evaluation and plotting module.
Handles metrics computation, confusion matrix, and training history plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from config import PLOT_NAME


def evaluate(model, X_train, y_train, X_val, y_val) -> None:
    """
    Prints a full evaluation report: metrics, prediction distribution,
    confusion matrix, and sklearn classification report.
    """
    # Predictions
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Losses
    train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]

    # Metrics
    metrics = {
        "train": _compute_metrics(y_train, y_train_pred, train_loss),
        "val": _compute_metrics(y_val, y_val_pred, val_loss),
    }

    # Print
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for split, m in metrics.items():
        label = "TRAIN" if split == "train" else "VALIDATION"
        print(f"\n📊 {label} METRICS:")
        print(f"   Loss:      {m['loss']:.4f}")
        print(f"   Accuracy:  {m['accuracy']:.4f}")
        print(f"   Precision: {m['precision']:.4f}")
        print(f"   Recall:    {m['recall']:.4f}")

    overfitting = metrics["train"]["accuracy"] - metrics["val"]["accuracy"]
    print(f"\n📈 Overfitting gap: {overfitting:.4f}")

    # Prediction distribution
    unique, counts = np.unique(y_val_pred, return_counts=True)
    print(f"\n🔍 PREDICTION DISTRIBUTION (val):")
    for cls, cnt in zip(unique, counts):
        name = "NORMAL" if cls == 0 else "PNEUMONIA"
        print(f"   {name}: {cnt} ({cnt / len(y_val_pred) * 100:.1f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"               NORMAL  PNEUMONIA")
    print(f"Actual NORMAL     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"       PNEUMONIA  {cm[1][0]:4d}     {cm[1][1]:4d}")

    print("\n" + classification_report(
        y_val, y_val_pred, target_names=["NORMAL", "PNEUMONIA"]
    ))


def plot_training_history(history, filename: str = PLOT_NAME) -> None:
    """
    Saves and displays accuracy + loss plots from a Keras History object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, loss) -> dict:
    return {
        "loss": loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }
