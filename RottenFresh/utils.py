import os
from pathlib import Path
from turtle import pd
from turtle import pd
from dotenv import load_dotenv

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_environment() -> None:
    load_dotenv()

def get_dataset_path() -> Path:
    load_environment()
    dataset = os.getenv("DATASET_PATH")

    if not dataset:
        raise ValueError("Please set DATASET_PATH environment variable")

    dataset = Path(dataset)

    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found:\n{dataset}")
    
    return dataset
    
    
    
def find_images(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "class_name": path.parent.name.lower()
            })
    return pd.DataFrame(rows)


def parse_class_name(class_name: str):
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
    print("Class counts:")
    print(df["class_name"].value_counts().rename_axis("class_name").reset_index(name="count"))

    print("\nProduce counts:")
    print(df["produce"].value_counts(dropna=False).rename_axis("produce").reset_index(name="count"))

    print("\nQuality label counts:")
    print(df["quality_label"].value_counts(dropna=False).rename_axis("quality_label").reset_index(name="count"))

    print("\nFreshness counts:")
    print(df["freshness"].value_counts(dropna=False).rename_axis("freshness").reset_index(name="count"))

if __name__ == "__main__":
    dataset = get_dataset_path()

    df = find_images(dataset)
    print(f"Found {len(df)} images")

    
    df[["produce", "quality_label", "freshness"]] = df["class_name"].apply(parse_class_name)
    
    #print(df.head())
    
    output_counts(df)