"""
Main training script.
Entry point: run this file to train the pneumonia detection model.

    python train.py
"""

from config import DATASET_PATH, TEST_PATH, BATCH_SIZE, MAX_EPOCHS, MODEL_NAME
from data_loader import load_dataset, compute_class_weights
from model import build_model
from callbacks import get_callbacks
from evaluate import evaluate, plot_training_history


def main():
    print("=" * 70)
    print("PNEUMONIA DETECTION — TRAINING")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_val, y_val = load_dataset(DATASET_PATH, TEST_PATH)

    # 2. Class weights
    class_weights = compute_class_weights(y_train)

    # 3. Build model
    print("\n[4/6] Building model...")
    model = build_model()
    model.summary()

    # 4. Train
    print("\n[5/6] Training...")
    print(f"   Epochs: {MAX_EPOCHS} | Batch size: {BATCH_SIZE} | LR: 0.001 (decay 0.95/epoch)")

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # 5. Evaluate
    print("\n[6/6] Evaluating...")
    evaluate(model, X_train, y_train, X_val, y_val)

    # 6. Plot
    plot_training_history(history)

    print(f"\n💾 Best model saved: {MODEL_NAME}")
    print("=" * 70)
    print("✅ DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
