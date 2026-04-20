import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_environment() -> None:
    """
    load environment variables from .env file if it exists
    """
    load_dotenv()


def get_dataset_path() -> Path:
    """
    Get the dataset path from the environment variable DATASET_PATH.
    """
    load_environment()
    dataset = os.getenv("DATASET_PATH")

    if not dataset:
        raise ValueError("Please set DATASET_PATH environment variable")

    dataset = Path(dataset)

    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found:\n{dataset}")

    return dataset


def find_images(root: Path) -> pd.DataFrame:
    """
    Find all image files under the given root directory and return a DataFrame with their paths, filenames, and class names.
    args:
        root: The root directory to search for images.
    returns:
        A DataFrame with columns: "path", "filename", "class_name".
    """
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "class_name": path.parent.name.lower(),
                }
            )
    return pd.DataFrame(rows)


def parse_class_name(class_name: str) -> None:
    """
    Parse the class name into produce, quality label, and freshness.
    args:
        class_name: The class name string
    """

    parts = class_name.split("__")

    produce = parts[0].strip().lower() if len(parts) > 0 else None
    quality_label = parts[1].strip().lower() if len(parts) > 1 else None

    # Map to a simple freshness label
    if quality_label == "healthy":
        freshness = "fresh"
    elif quality_label == "rotten":
        freshness = "rotten"
    else:
        freshness = None

    return pd.Series([produce, quality_label, freshness])


def output_counts(df: pd.DataFrame) -> None:
    """
    Print counts of each class, produce, quality label, and freshness in the dataset.
    args:
        df: The DataFrame containing the dataset information.
    """
    print("Class counts:")
    print(
        df["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
    )

    print("\nProduce counts:")
    print(
        df["produce"]
        .value_counts(dropna=False)
        .rename_axis("produce")
        .reset_index(name="count")
    )

    print("\nQuality label counts:")
    print(
        df["quality_label"]
        .value_counts(dropna=False)
        .rename_axis("quality_label")
        .reset_index(name="count")
    )

    print("\nFreshness counts:")
    print(
        df["freshness"]
        .value_counts(dropna=False)
        .rename_axis("freshness")
        .reset_index(name="count")
    )


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from the given path and converts it to RGB format.
    args:
        path: The path to the image file.
    returns:
        The image in RGB format as a NumPy array.
    """

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def show_sample_images(sample_df, n=6, title=None) -> None:
    """
    Shows a sample of images from the dataset with their class names as titles.
    args:
        sample_df: A DataFrame containing at least "path" and "class_name" columns.
        n: The number of sample images to display.
        title: An optional title for the entire figure.
    """

    if len(sample_df) == 0:
        print("No images to display.")
        return

    sample_df = sample_df.sample(min(n, len(sample_df)), random_state=42)

    fig, axes = plt.subplots(1, len(sample_df), figsize=(4 * len(sample_df), 4))
    if len(sample_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img = load_image(row["path"])
        ax.imshow(img)
        ax.set_title(row["class_name"])
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_image_shape(path: str) -> tuple:
    """
    Gets the shape of an image from the given path.
    args:
        path: The path to the image file.
    returns:
        A tuple (height, width, channels) representing the shape of the image.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None, None, None
    h, w, c = img.shape
    return h, w, c


if __name__ == "__main__":
    dataset = get_dataset_path()

    df = find_images(dataset)
    print(f"Found {len(df)} images")

    df[["produce", "quality_label", "freshness"]] = df["class_name"].apply(
        parse_class_name
    )
    healthy_df = df[df["quality_label"] == "healthy"]
    rotten_df = df[df["quality_label"] == "rotten"]

    # outputs:
    # print(df.head())
    # output_counts(df)
    # show_sample_images(df, n=6, title="Sample Images from Dataset")

    # show_sample_images(healthy_df, n=4, title=f"Healthy {FOCUS_PRODUCE} examples")
    # show_sample_images(rotten_df, n=4, title=f"Rotten {FOCUS_PRODUCE} examples")
