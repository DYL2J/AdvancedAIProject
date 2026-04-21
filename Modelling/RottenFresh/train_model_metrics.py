import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class Config:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0

    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4

    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    target_column: str = "freshness"
    produce_column: str = "produce"

    checkpoint_path: str = "ResNet18Model_Checkpoint.pth"
    final_model_path: str = "ResNet18Model_Final.pth"
    history_csv_path: str = "training_history.csv"

    train_split_csv: str = "train_split.csv"
    val_split_csv: str = "val_split.csv"
    test_split_csv: str = "test_split.csv"
    test_predictions_csv: str = "test_predictions_resnet18_noleak.csv"
    produce_metrics_csv: str = "produce_class_metrics.csv"

    dataset_csv: str = "deduplicated_dataset.csv"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clean_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the deduplicated dataset CSV.

    args:
        csv_path: Path to the dataset CSV file.

    returns:
        A cleaned pandas DataFrame.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)

    required_columns = {
        "path",
        CFG.target_column,
        CFG.produce_column,
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Expected columns missing from dataset: {sorted(missing_columns)}"
        )

    df["path"] = df["path"].astype(str)
    df[CFG.target_column] = df[CFG.target_column].astype(str).str.lower().str.strip()
    df[CFG.produce_column] = df[CFG.produce_column].astype(str).str.lower().str.strip()

    df = df[df["path"].notna()].copy()
    df = df[df[CFG.target_column].notna()].copy()
    df = df[df[CFG.produce_column].notna()].copy()

    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    missing_files = [p for p in df["path"] if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Found {len(missing_files)} missing image paths in the CSV. "
            f"Example: {missing_files[0]}"
        )

    return df.reset_index(drop=True)


class ProduceDataset(Dataset):
    """
    Custom dataset for loading produce images and labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        args:
            df: DataFrame containing image paths and labels.
            class_to_idx: Mapping from class name to integer index.
            transform: Optional torchvision transform pipeline.
        """
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load an image and label, apply transforms, and return tensors.

        args:
            idx: The index of the sample to retrieve.

        returns:
            A tuple of (image_tensor, label_index).
        """
        row = self.df.iloc[idx]

        image_path = row["path"]
        label_name = row[CFG.target_column]

        with Image.open(image_path) as img:
            image = img.convert("RGBA").convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[label_name]
        return image, label


train_transform = transforms.Compose(
    [
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def build_splits(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    dict[str, int],
    dict[int, str],
]:
    """
    Split dataset into train, validation, and test sets.

    args:
        df: Full dataset DataFrame.

    returns:
        train_df, val_df, test_df, class_names, class_to_idx, idx_to_class
    """
    class_names = sorted(df[CFG.target_column].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    stratify_labels = (
        df[CFG.produce_column].astype(str)
        + "__"
        + df[CFG.target_column].astype(str)
    )

    train_df, holdout_df = train_test_split(
        df,
        test_size=CFG.val_size + CFG.test_size,
        stratify=stratify_labels,
        random_state=CFG.random_state,
    )

    relative_test_size = CFG.test_size / (CFG.val_size + CFG.test_size)

    holdout_stratify_labels = (
        holdout_df[CFG.produce_column].astype(str)
        + "__"
        + holdout_df[CFG.target_column].astype(str)
    )

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        stratify=holdout_stratify_labels,
        random_state=CFG.random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(CFG.train_split_csv, index=False)
    val_df.to_csv(CFG.val_split_csv, index=False)
    test_df.to_csv(CFG.test_split_csv, index=False)

    return train_df, val_df, test_df, class_names, class_to_idx, idx_to_class


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for train, validation, and test sets.
    """
    train_dataset = ProduceDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = ProduceDataset(val_df, class_to_idx, transform=eval_transform)
    test_dataset = ProduceDataset(test_df, class_to_idx, transform=eval_transform)

    pin_memory = CFG.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_model(num_classes: int) -> nn.Module:
    """
    Build and return a ResNet18 model adapted for classification.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(CFG.device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy for a batch.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for images, labels in loader:
        images = images.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += compute_accuracy(logits, labels) * batch_size
        total_n += batch_size

    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluate the model on validation data.
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for images, labels in loader:
        images = images.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += compute_accuracy(logits, labels) * batch_size
        total_n += batch_size

    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[list[int], list[int], list[list[float]]]:
    """
    Run inference over a dataloader.

    returns:
        y_true, y_pred, y_probs
    """
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_probs: list[list[float]] = []

    for images, labels in loader:
        images = images.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())

    return y_true, y_pred, y_probs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Train the model and keep the best checkpoint by validation loss.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, CFG.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoch {epoch}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CFG.checkpoint_path)
            print(f"Saved best checkpoint to {CFG.checkpoint_path}")

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(CFG.history_csv_path, index=False)
    print(f"Saved training history to {CFG.history_csv_path}")

    return model, history_df


def save_produce_class_metrics(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute and save freshness prediction performance for each produce class.

    args:
        predictions_df: DataFrame containing true and predicted labels.

    returns:
        A DataFrame of metrics per produce class.
    """
    metric_rows: list[dict[str, float | int | str]] = []

    for produce_class, group_df in predictions_df.groupby(CFG.produce_column):
        accuracy = accuracy_score(
            group_df["true_label"],
            group_df["predicted_label"],
        )
        macro_f1 = f1_score(
            group_df["true_label"],
            group_df["predicted_label"],
            average="macro",
            zero_division=0,
        )

        metric_rows.append(
            {
                CFG.produce_column: produce_class,
                "sample_count": len(group_df),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=CFG.produce_column
    ).reset_index(drop=True)

    metrics_df.to_csv(CFG.produce_metrics_csv, index=False)
    print(f"Saved produce class metrics to {CFG.produce_metrics_csv}")

    return metrics_df


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    test_df: pd.DataFrame,
    idx_to_class: dict[int, str],
) -> None:
    """
    Evaluate the trained model on the test set and save predictions.
    """
    y_true, y_pred, y_probs = predict_loader(model, test_loader)

    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    y_true_names = [idx_to_class[i] for i in y_true]
    y_pred_names = [idx_to_class[i] for i in y_pred]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\nTEST RESULTS")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    predictions_df = test_df.copy()
    predictions_df["true_label"] = y_true_names
    predictions_df["predicted_label"] = y_pred_names
    predictions_df["is_correct"] = (
        predictions_df["true_label"] == predictions_df["predicted_label"]
    )

    for class_idx, class_name in idx_to_class.items():
        predictions_df[f"prob_{class_name}"] = [
            probs[class_idx] for probs in y_probs
        ]

    predictions_df.to_csv(CFG.test_predictions_csv, index=False)
    print(f"Saved test predictions to {CFG.test_predictions_csv}")

    produce_metrics_df = save_produce_class_metrics(predictions_df)

    print("\nPER-PRODUCE-CLASS RESULTS")
    print("-" * 50)
    print(produce_metrics_df.to_string(index=False))


def main() -> None:
    """
    Run the full training pipeline using deduplicated_dataset.csv.
    """
    set_seed(CFG.random_state)

    print(f"Using device: {CFG.device}")
    print(f"Loading dataset from: {CFG.dataset_csv}")

    df = load_clean_dataframe(CFG.dataset_csv)
    print(f"Loaded {len(df)} rows from deduplicated dataset")

    train_df, val_df, test_df, class_names, class_to_idx, idx_to_class = build_splits(
        df
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")
    print(f"Test size : {len(test_df)}")
    print(f"Classes   : {class_names}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        class_to_idx,
    )

    model = build_model(len(class_names))
    model, _ = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), CFG.final_model_path)
    print(f"Saved final model to {CFG.final_model_path}")

    evaluate_on_test(model, test_loader, test_df, idx_to_class)


if __name__ == "__main__":
    main()