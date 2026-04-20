# Produce Freshness & Quality Analysis

This project provides a complete computer vision pipeline for analyzing produce images. It combines deep learning classification, classical feature extraction, explainability (SHAP), and rule-based quality scoring.

The system supports:
- Fresh vs. rotten classification using a trained ResNet18 model
- Defect localization using SHAP explanations
- Feature-based quality inspection (color, texture, shape)
- Dataset cleaning and deduplication
- Model training and evaluation workflows


## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Analysis Pipeline](#analysis-pipeline)
- [Outputs](#outputs)
- [Notes](#notes)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure dataset path

### 3. Create a .env file:
This .env is based off the .env_example file
```env
DATASET_PATH=/absolute/path/to/your/dataset
```

#### Example Dataset Structure:
```bash
dataset/
├── apple__healthy/
├── apple__rotten/
├── banana__healthy/
├── banana__rotten/
```


## Project Structure



## Table of Contents

- Setup
- Project Structure
- Usage
- Data Pipeline
- Model Training
- Analysis Pipeline
- Outputs
- Notes

---

## Setup

### 1. Clone the repository

    git clone <your-repo-url>
    cd <repo-name>

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Configure dataset path

Create a `.env` file:

    DATASET_PATH=/absolute/path/to/your/dataset

Example dataset structure:

    dataset/
    ├── apple__healthy/
    ├── apple__rotten/
    ├── banana__healthy/
    ├── banana__rotten/

---

## Project Structure

### utils.py

Provides dataset handling and utility functions:
- get_dataset_path() – Reads dataset path from `.env`
- find_images() – Recursively finds image files
- parse_class_name() – Extracts produce, quality label, and freshness
- load_image() – Loads image as RGB
- show_sample_images() – Visualization helper
- output_counts() – Dataset statistics

---

### clean_dataset.py

Removes duplicate images and prepares dataset CSV.

Key functions:
- build_dataframe()
- file_md5()
- remove_exact_duplicates()
- check_cross_label_conflicts()
- purge_duplicates()

Output:

    deduplicated_dataset.csv

---

### feature_extract.py

Extracts interpretable features from images.

Segmentation:
- grabcut_mask()
- apply_mask()
- crop_to_mask()

Feature groups:

Shape:
- area ratio
- solidity
- circularity
- extent
- perimeter

Color:
- RGB and HSV means
- HSV standard deviation
- dark ratio (decay proxy)
- brown ratio (rot proxy)

Texture:
- Laplacian variance
- grayscale standard deviation

Core function:
- extract_features_for_image()

Visualization:
- show_sample_segmentations()
- plot_feature_distributions()

---

### train_model.py

Trains a ResNet18 classifier using a cleaned dataset CSV.

Key components:
- Config
- ProduceDataset
- build_splits()
- build_dataloaders()
- build_model()
- train_one_epoch()
- validate_one_epoch()

Outputs:
- Model checkpoints
- Final model
- Training history CSV
- Dataset splits
- Test predictions

---

### resnet_loso.py

Leave-One-Produce-Out (LOSO) evaluation for generalization testing.

Key features:
- Holds out one produce class for testing
- Trains on remaining classes
- Evaluates cross-produce performance

Important functions:
- build_fold_dataframes()
- build_dataloaders()
- build_model()

---

### product_analysis.py

End-to-end pipeline combining:
- Model inference
- SHAP explainability
- Feature-based quality scoring

Main class:
- ProductAnalyzer

Key methods:

Model:
- _load_model()
- predict_from_rgb_numpy()

Freshness:
- run_freshness_evaluation()
- _save_freshness_result()

Defect Detection:
- run_defect_detection()
- _make_defect_overlay()

Quality Inspection:
- run_quality_inspection()
- _grade_quality()

Full Pipeline:
- analyze()

Wrapper:
- analyze_product()

Includes CLI support.

---

### demo_file.py

Minimal example:

    from product_analysis import analyze_product

    result = analyze_product(
        image_path="rotten_apple.jpg",
        checkpoint_path="FoodModel_1.pth",
        output_dir="analysis_outputs",
    )

    print(result)

---

### .env_example

    DATASET_PATH=C:\Your_Path

---

### .gitignore

    .env
    deduplicated_dataset.csv

---

## Usage

### 1. Clean dataset

    python clean_dataset.py

### 2. Train model

    python train_model.py

### 3. Run LOSO evaluation
For evaluation of generalisation only.

    python resnet_loso.py

### 4. Analyze an image

    python product_analysis.py path/to/image.jpg --checkpoint model.pth

---

## Data Pipeline

1. Raw dataset (folder structure)
2. utils.py → scan + parse labels
3. clean_dataset.py → deduplicate
4. train_model.py → train classifier
5. product_analysis.py → inference + analysis

---

## Analysis Pipeline

For each image:

1. Load and preprocess image
2. Predict freshness (ResNet18)
3. Generate SHAP explanation
4. Identify defect regions
5. Segment object (GrabCut)
6. Extract features (shape, color, texture)
7. Compute rule-based quality score
8. Save outputs

---


## Outputs

Generated per image:

    analysis_outputs/<image_name>/
    ├── freshness_prediction.png
    ├── shap_explanation.png
    ├── defect_highlight_overlay.png
    ├── defect_mask.png
    ├── segmented_product.png
    ├── quality_features.json
    ├── analysis_summary.json

---


## Notes
Dataset labels must follow format:

    produce__quality

Example:

      apple__healthy
      apple__rotten


Feature-based grading is heuristic and configurable in _grade_quality()

Improvement suggestion: Add config file (YAML/JSON)