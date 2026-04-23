import kagglehub
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional: custom download location
DATASET_PATH = os.getenv("DATASET_PATH", "./dataset")

print("Downloading dataset...")

# Download dataset
path = kagglehub.dataset_download(
    "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten",
    path=DATASET_PATH
)

print(f"Dataset downloaded to: {path}")