"""Dataset and preprocessing utilities."""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import torch


class ImageFolderFromSamples(Dataset):
    """Dataset wrapper for (path, label) samples.

    args:
        samples: List of (image_path, label) tuples.
        transform: Optional torchvision transform pipeline.

    returns:
        A PyTorch Dataset that loads images from file paths and returns
        transformed image-label pairs.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.

        returns:
            The total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a single image-label pair.

        args:
            idx: Index of the sample to retrieve.

        returns:
            A tuple containing the transformed image and its label.
        """
        path, target = self.samples[idx]

        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target


def compute_mean_std(samples, img_size, num_workers, sample_limit=4000):
    """Compute dataset mean/std.

    args:
        samples: List of (image_path, label) tuples.
        img_size: Target image size used before converting to tensors.
        num_workers: Number of DataLoader worker processes.
        sample_limit: Maximum number of samples to use when estimating
            channel statistics.

    returns:
        A tuple of (mean, std), where each is a list of three floats
        representing the per-channel RGB statistics.
    """
    if len(samples) > sample_limit:
        indices = np.random.choice(len(samples), sample_limit, replace=False)
        samples = [samples[i] for i in indices]

    dataset = ImageFolderFromSamples(
        samples,
        transform=transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0

    for x, _ in tqdm(loader, desc="Mean/Std", leave=False):
        b = x.size(0)
        x = x.view(b, 3, -1)

        mean += x.mean(dim=2).sum(dim=0)
        std += x.std(dim=2).sum(dim=0)
        count += b

    mean /= count
    std /= count

    return mean.tolist(), std.tolist()


def get_transforms(mean, std, img_size):
    """Return train and eval transforms.

    args:
        mean: Per-channel RGB mean values used for normalisation.
        std: Per-channel RGB standard deviation values used for
            normalisation.
        img_size: Target square image size.

    returns:
        A tuple of (train_tfms, eval_tfms), where each item is a
        torchvision transform pipeline.
    """
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.08)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tfms, eval_tfms
