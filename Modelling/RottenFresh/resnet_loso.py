import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import Modelling.RottenFresh.utils as utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Config:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0

    epochs: int = (
        3  # smaller to reduce training time for this test, increase for better performance
    )
    lr: float = 1e-4
    weight_decay: float = 1e-4

    use_pretrained: bool = True
    freeze_backbone: bool = False

    target_column: str = "freshness"
    group_column: str = "produce"
    allowed_targets: list[str] = ["fresh", "rotten"]

    val_size_within_train: float = 0.15
    random_state: int = 42
    stratify: bool = True

    output_dir: str = "loso_produce_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    args:
        seed: The random seed value.
    returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataframe() -> pd.DataFrame:
    """
    Build dataset dataframe using utils helpers.
    returns:
        Clean dataframe with labels and metadata.
    """
    dataset_path = utils.get_dataset_path()
    df = utils.find_images(dataset_path).copy()

    if df.empty:
        raise ValueError(f"No images found under dataset path: {dataset_path}")

    df[["produce", "quality_label", "freshness"]] = df["class_name"].apply(
        utils.parse_class_name
    )

    if CFG.target_column not in df.columns:
        raise ValueError(f"Target column '{CFG.target_column}' not found")
    if CFG.group_column not in df.columns:
        raise ValueError(f"Group column '{CFG.group_column}' not found")

    df = df[df[CFG.target_column].notna()].copy()
    df = df[df[CFG.group_column].notna()].copy()

    if CFG.allowed_targets:
        df = df[df[CFG.target_column].isin(CFG.allowed_targets)].copy()
    if df.empty:
        raise ValueError("No usable rows remain after filtering")

    return df.reset_index(drop=True)


class ProduceImageDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, class_to_idx: dict[str, int], transform=None
    ) -> None:
        """
        args:
            df: DataFrame containing image paths and labels.
            class_to_idx: Mapping from class name to index.
            transform: Optional transform pipeline.
        returns:
            None
        """
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        args:
            idx: Index of the sample.
        returns:
            Tuple (image_tensor, label_index).
        """
        row = self.df.iloc[idx]
        image_path = row["path"]
        label_name = row[CFG.target_column]

        with Image.open(image_path) as img:
            image = img.convert("RGBA").convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label_name]


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


def make_output_dir() -> Path:
    """
    Create output directory if needed.
    returns:
        Path to output directory.
    """
    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_class_mappings(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Generate class mappings.
    args:
        df: Dataset dataframe.
    returns:
        class_names, class_to_idx, idx_to_class
    """
    class_names = sorted(df[CFG.target_column].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    return class_names, class_to_idx, idx_to_class


def build_fold_dataframes(
    df: pd.DataFrame, held_out_produce: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build LOSO split.
    args:
        df: Full dataset dataframe.
        held_out_produce: Produce class to hold out.
    returns:
        train_df, val_df, test_df
    """
    test_df = df[df[CFG.group_column] == held_out_produce].copy()
    trainval_df = df[df[CFG.group_column] != held_out_produce].copy()

    if test_df.empty:
        raise ValueError(f"No rows for held-out produce: {held_out_produce}")
    if trainval_df.empty:
        raise ValueError(f"No training rows remain after holdout: {held_out_produce}")

    stratify_col = trainval_df[CFG.target_column] if CFG.stratify else None
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=CFG.val_size_within_train,
        random_state=CFG.random_state,
        stratify=stratify_col,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders for training, validation, and test.
    args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        test_df: Test dataframe.
        class_to_idx: Class mapping.
    returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = ProduceImageDataset(
        train_df, class_to_idx, transform=train_transform
    )
    val_dataset = ProduceImageDataset(val_df, class_to_idx, transform=eval_transform)
    test_dataset = ProduceImageDataset(test_df, class_to_idx, transform=eval_transform)

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
    Build ResNet18 model.
    args:
        num_classes: Number of output classes.
    returns:
        Model instance.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if CFG.use_pretrained else None
    )

    if CFG.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True
    return model.to(CFG.device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy.
    args:
        logits: Model outputs.
        labels: Ground truth labels.
    returns:
        Accuracy value.
    """
    return (torch.argmax(logits, dim=1) == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    """
    Train model for one epoch.
    returns:
        loss, accuracy
    """
    model.train()
    total_loss = total_acc = total_n = 0

    for images, labels in loader:
        images, labels = images.to(CFG.device, non_blocking=True), labels.to(
            CFG.device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += compute_accuracy(logits, labels) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    """
    Validate model for one epoch.
    returns:
        loss, accuracy
    """
    model.eval()
    total_loss = total_acc = total_n = 0

    for images, labels in loader:
        images, labels = images.to(CFG.device, non_blocking=True), labels.to(
            CFG.device, non_blocking=True
        )
        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += compute_accuracy(logits, labels) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n
