import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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

    checkpoint_path: str = "ResNet18Model_Checkpoint.pth"
    final_model_path: str = "ResNet18Model_Final.pth"
    history_csv_path: str = "training_history.csv"

    train_split_csv: str = "train_split.csv"
    val_split_csv: str = "val_split.csv"
    test_split_csv: str = "test_split.csv"
    test_predictions_csv: str = "test_predictions_resnet18_noleak.csv"

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
    df: pd.DataFrame = pd.read_csv(csv_path)

    if "file_hash" in df.columns:
        df = df.drop(columns=["file_hash"])

    if "path" not in df.columns:
        raise ValueError("Expected a 'path' column in dataset")

    if CFG.target_column not in df.columns:
        raise ValueError(f"Expected target column '{CFG.target_column}' in dataset")

    df["path"] = df["path"].astype(str)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


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
        self.df: pd.DataFrame = df.reset_index(drop=True).copy()
        self.class_to_idx: dict[str, int] = class_to_idx
        self.transform: transforms.Compose | None = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        returns:
            The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Loads an image and its corresponding label, applies transforms, and returns them as tensors.
        args:
            idx: The index of the sample to retrieve.
        returns:
            A tuple (image_tensor, label_index) where:
                - image_tensor: The transformed image as a PyTorch tensor.
                - label_index: The integer index of the class label.
        """
        row: pd.Series = self.df.iloc[idx]

        image_path: str = row["path"]
        label_name: str = row[CFG.target_column]

        with Image.open(image_path) as img:
            image = img.convert("RGBA").convert("RGB")

        if self.transform:
            image = self.transform(image)

        label: int = self.class_to_idx[label_name]
        return image, label


# Transofrmers
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
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, int], dict[int, str]
]:
    """
    Split dataset into train, validation, and test sets.
    args:
        df: Full dataset DataFrame.
    returns:
        train_df, val_df, test_df, class_names, class_to_idx, idx_to_class
    """
    class_names: list[str] = sorted(df[CFG.target_column].unique().tolist())
    class_to_idx: dict[str, int] = {name: i for i, name in enumerate(class_names)}
    idx_to_class: dict[int, str] = {i: name for name, i in class_to_idx.items()}

    train_df, holdout_df = train_test_split(
        df,
        test_size=CFG.val_size + CFG.test_size,
        stratify=df[CFG.target_column],
        random_state=CFG.random_state,
    )

    relative_test_size: float = CFG.test_size / (CFG.val_size + CFG.test_size)

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        stratify=holdout_df[CFG.target_column],
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

    pin_memory: bool = CFG.device == "cuda"

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
    model: nn.Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(CFG.device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy for a batch.
    """
    preds: torch.Tensor = torch.argmax(logits, dim=1)
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

    total_loss, total_acc, total_n = 0.0, 0.0, 0

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
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    """
    Evaluate the model on validation data.
    """
    model.eval()

    total_loss, total_acc, total_n = 0.0, 0.0, 0

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


def main() -> None:
    set_seed(CFG.random_state)

    df = load_clean_dataframe(CFG.dataset_csv)

    train_df, val_df, test_df, class_names, class_to_idx, idx_to_class = build_splits(df)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, class_to_idx
    )

    model = build_model(len(class_names))

    print("Data loaded and model initialized.")


if __name__ == "__main__":
    main()