import hashlib

import pandas as pd

import Modelling.RottenFresh.utils as utils

# config
TARGET_COLUMN = "freshness"
ALLOWED_TARGETS = ["fresh", "rotten"]

OUTPUT_CSV = "deduplicated_dataset.csv"


def file_md5(path: str, chunk_size: int = 8192) -> str:
    """
    Compute the MD5 hash of a file.
    args:
        path: Path to the file to hash.
        chunk_size: Size of chunks to read the file in (default: 8192 bytes
    returns:
        The hexadecimal MD5 hash of the file.
    """

    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Build dataframe (same as your training code)
def build_dataframe() -> pd.DataFrame:
    dataset_path = utils.get_dataset_path()
    df = utils.find_images(dataset_path).copy()

    if df.empty:
        raise ValueError(f"No images found under dataset path: {dataset_path}")

    df[["produce", "quality_label", "freshness"]] = df["class_name"].apply(
        utils.parse_class_name
    )

    df = df[df[TARGET_COLUMN].notna()].copy()

    if ALLOWED_TARGETS is not None:
        df = df[df[TARGET_COLUMN].isin(ALLOWED_TARGETS)].copy()

    df["path"] = df["path"].astype(str)
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate images based on file hash.
        - This will keep the first occurrence of each unique image and drop subsequent duplicates.
    args:
        df: DataFrame with a "path" column pointing to image files.
    returns:
        A deduplicated DataFrame with an additional "file_hash" column.
    """

    df = df.copy()

    print("\nHashing all images (this may take a minute)...")
    df["file_hash"] = df["path"].apply(file_md5)

    before = len(df)

    # Keep first occurrence of each unique image
    df_dedup = df.drop_duplicates(subset=["file_hash"]).reset_index(drop=True)

    after = len(df_dedup)
    removed = before - after

    print("\n" + "=" * 50)
    print("DEDUPLICATION SUMMARY")
    print("=" * 50)
    print(f"Original images : {before}")
    print(f"Unique images   : {after}")
    print(f"Removed dupes   : {removed}")
    print(f"Reduction       : {removed / before:.2%}")

    return df_dedup


def check_cross_label_conflicts(df: pd.DataFrame) -> None:
    """
    Check for cases where the same image (same file hash) has multiple freshness labels.
     - This can happen if the same image is duplicated in different folders with different labels.
    args:
        df: DataFrame with columns "file_hash" and TARGET_COLUMN (freshness)
    """

    print("\nChecking for duplicate images with conflicting labels")

    label_counts = df.groupby("file_hash")[TARGET_COLUMN].nunique()
    bad_hashes = label_counts[label_counts > 1].index

    if len(bad_hashes) == 0:
        print("No cross label duplicates found")
        return

    print(f"WARNING: {len(bad_hashes)} hashes have multiple labels")

    bad_rows = df[df["file_hash"].isin(bad_hashes)].sort_values("file_hash")
    print(bad_rows[["path", TARGET_COLUMN]].head(20))


def purge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate images and checks for cross-label conflicts.
    This is a convenience function that combines the deduplication and conflict checking steps.
    args:
        df: DataFrame with a "path" column and TARGET_COLUMN (freshness)
    returns:
        df: A deduplicated DataFrame with an additional "file_hash" column, and prints any cross-label conflicts.
    """
    df_dedup = remove_exact_duplicates(df)
    check_cross_label_conflicts(df_dedup)
    return df_dedup


def main() -> None:
    df_dedup = purge_duplicates(build_dataframe())
    df_dedup.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved deduplicated dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
