"""Main training script."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import datasets

from config import *
from dataset import (
    ImageFolderFromSamples,
    compute_mean_std,
    get_transforms,
)
from engine import train_one_epoch, evaluate
from model import UResNetClassifier


def main():
    """Run the full training and evaluation pipeline.

    returns:
        None. Trains the model, saves the best checkpoint, and prints the
        final test F1 score.
    """
    print(f"Using device: {DEVICE}")

    root = Path(DATASET_ROOT)
    dataset = datasets.ImageFolder(root)

    samples = dataset.samples
    targets = [t for _, t in samples]

    indices = list(range(len(samples)))

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=targets,
        random_state=SEED,
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=[targets[i] for i in trainval_idx],
        random_state=SEED,
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    if COMPUTE_MEAN_STD:
        mean, std = compute_mean_std(
            train_samples,
            IMG_SIZE,
            NUM_WORKERS,
        )
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    train_tfms, eval_tfms = get_transforms(mean, std, IMG_SIZE)

    train_ds = ImageFolderFromSamples(train_samples, train_tfms)
    val_ds = ImageFolderFromSamples(val_samples, eval_tfms)
    test_ds = ImageFolderFromSamples(test_samples, eval_tfms)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
    )

    model = UResNetClassifier(len(dataset.classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            DEVICE,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_CHECKPOINT)

        print(f"Time: {time.time() - start:.1f}s")

    model.load_state_dict(torch.load(BEST_CHECKPOINT))

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
