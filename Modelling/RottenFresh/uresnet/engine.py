"""Training and evaluation logic for the U-ResNet classifier."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train the model for one epoch.

    args:
        model: The neural network being trained.
        loader: DataLoader providing training batches.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model weights.
        scaler: Gradient scaler for mixed precision training.
        device: Device string, typically "cpu" or "cuda".

    returns:
        A tuple of (loss, accuracy, macro_f1) for the epoch.
    """
    model.train()

    losses = []
    predictions = []
    targets = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            enabled=(device == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    accuracy = np.mean(np.array(predictions) == np.array(targets))
    macro_f1 = f1_score(targets, predictions, average="macro")

    return float(np.mean(losses)), float(accuracy), float(macro_f1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on validation or test data.

    args:
        model: The neural network being evaluated.
        loader: DataLoader providing evaluation batches.
        criterion: Loss function used for evaluation.
        device: Device string, typically "cpu" or "cuda".

    returns:
        A tuple of (loss, accuracy, macro_f1, predictions, targets).
    """
    model.eval()

    losses = []
    predictions = []
    targets = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            enabled=(device == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        losses.append(loss.item())
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(predictions) == np.array(targets))
    macro_f1 = f1_score(targets, predictions, average="macro")

    return (
        float(np.mean(losses)),
        float(accuracy),
        float(macro_f1),
        np.array(predictions),
        np.array(targets),
    )
