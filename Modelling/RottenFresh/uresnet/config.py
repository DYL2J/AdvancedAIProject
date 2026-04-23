"""Configuration module for UResNet training.
This file centralises all hyperparameters and paths so that experiments
can be modified without touching the training logic.
"""

import torch
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT", "PATH_TO_DATASET")

IMG_SIZE = 192
BATCH_SIZE = 48
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 6
SEED = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15

COMPUTE_MEAN_STD = True
USE_WEIGHTED_SAMPLER = False
MODEL_USE_EUCB = False

BEST_CHECKPOINT = "best_uresnet.pth"
LATEST_CHECKPOINT = "latest_uresnet.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
