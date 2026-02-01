"""
Configuration file - modify parameters here.
"""

# Paths
DATASET_PATH = "chest_xray/train"
TEST_PATH = "chest_xray/test"
MODEL_NAME = "pneumonia_model.keras"
PLOT_NAME = "training_history.png"

# Image
IMG_SIZE = 150

# Training
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
LR_DECAY = 0.95  # Per epoch: lr * (LR_DECAY ^ epoch)

# Regularization
DROPOUT_RATES = [0.2, 0.2, 0.3, 0.3, 0.5, 0.5]  # Per ogni layer
L2_FACTOR = 0.001

# Augmentation
AUG_ROTATION = 0.05
AUG_ZOOM = 0.05
AUG_TRANSLATION = 0.05

# Callbacks
EARLY_STOP_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
