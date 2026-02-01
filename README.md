# Pneumonia Detection from Chest X-Rays

Binary image classification model that detects pneumonia from chest X-ray images using a CNN built with TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.10%2B-3572A5?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Dependencies

| Package | Min Version | Purpose |
|---|---|---|
| Python | 3.10 | Runtime |
| TensorFlow | 2.16 | Deep learning backend |
| tf-keras | 2.16 | Model API |
| NumPy | 1.23 | Array operations |
| OpenCV | 4.5 | Image loading and resizing |
| scikit-learn | 1.1 | Metrics and class weights |
| Matplotlib | 3.5 | Training plots |

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── config.py          # All hyperparameters in one place
│   ├── data_loader.py     # Image loading, normalization, class weights
│   ├── model.py           # CNN architecture + augmentation pipeline
│   ├── callbacks.py       # Keras training callbacks
│   ├── evaluate.py        # Metrics, confusion matrix, plots
│   ├── train.py           # Main training script (entry point)
│   └── predict.py         # Inference on single images or folders
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/maccariluca/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
```

## Dataset

The project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

Download and place it so that the structure matches:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Download via Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

> Make sure `chest_xray/` is placed in the project root (next to `src/`).

## Training

```bash
cd src
python train.py
```

This will:
1. Load and preprocess the dataset
2. Build and compile the CNN
3. Train with early stopping and LR scheduling
4. Print a full evaluation report (accuracy, precision, recall, confusion matrix)
5. Save the best model as `pneumonia_model.keras`
6. Save the training plot as `training_history.png`

### Tuning Hyperparameters

All parameters live in `src/config.py`. No need to touch any other file:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 150 | Input image resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `MAX_EPOCHS` | 30 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Initial Adam learning rate |
| `LR_DECAY` | 0.95 | Per-epoch exponential LR decay |
| `DROPOUT_RATES` | [0.2, 0.2, 0.3, 0.3, 0.5, 0.5] | Dropout per layer |
| `EARLY_STOP_PATIENCE` | 8 | Epochs before early stop |

## Inference

Predict a single image:

```bash
cd src
python predict.py --image path/to/xray.jpeg
```

Predict all images in a folder:

```bash
python predict.py --folder chest_xray/test/NORMAL
```

## Model Architecture

```
Input (150×150×1)
    │
    ▼
┌─ Augmentation ────────────────────┐
│  RandomFlip · Rotation · Zoom     │
│  Translation                      │
└───────────────────────────────────┘
    │
    ▼
┌─ Conv Block 1 ────────────────────┐
│  Conv2D(16) → BatchNorm → ReLU   │
│  MaxPool → Dropout(0.2)          │
└───────────────────────────────────┘
    │
    ▼
┌─ Conv Block 2 ────────────────────┐
│  Conv2D(32) → BatchNorm → ReLU   │
│  MaxPool → Dropout(0.2)          │
└───────────────────────────────────┘
    │
    ▼
┌─ Conv Block 3 ────────────────────┐
│  Conv2D(64) → BatchNorm → ReLU   │
│  MaxPool → Dropout(0.3)          │
└───────────────────────────────────┘
    │
    ▼
┌─ Conv Block 4 ────────────────────┐
│  Conv2D(128) → BatchNorm → ReLU  │
│  MaxPool → Dropout(0.3)          │
└───────────────────────────────────┘
    │
    ▼
  Flatten
    │
    ▼
┌─ Dense 1 ─────────────────────────┐
│  Dense(256) → BatchNorm → ReLU   │
│  Dropout(0.5)                    │
└───────────────────────────────────┘
    │
    ▼
┌─ Dense 2 ─────────────────────────┐
│  Dense(128) → BatchNorm → ReLU   │
│  Dropout(0.5)                    │
└───────────────────────────────────┘
    │
    ▼
  Dense(2, softmax)
  [NORMAL | PNEUMONIA]
```

## Anti-Overfitting Techniques

| Technique | Where |
|---|---|
| Data augmentation | `model.py` — flip, rotation, zoom, translation |
| Batch normalization | After every Conv2D and Dense layer |
| Dropout (progressive) | 0.2 → 0.3 → 0.5 through the network |
| Balanced class weights | `data_loader.py` — sqrt-moderated weights |
| Early stopping | `callbacks.py` — patience 8, monitors val_accuracy |
| LR decay | Exponential: `lr × 0.95^epoch` |
| ReduceLROnPlateau | Halves LR when val_loss stalls |
