# U-ResNet for Fruit and Vegetable Freshness Classification

## Overview

This module implements a **U-ResNet image classifier** for identifying the freshness state of fruit and vegetable images.

The model is designed for a **multi-class classification task**, where each class combines both the produce type and its condition, for example:

- `Apple__Healthy`
- `Apple__Rotten`
- `Banana__Healthy`
- `Banana__Rotten`

The code is structured to support reproducible training, validation, and testing using a clean PyTorch pipeline.

## Purpose

The goal of this implementation is to classify produce images into the correct freshness category by combining:

- a **ResNet branch** for strong discriminative feature learning
- a **U-Net branch** for richer spatial and hierarchical feature extraction

These two branches are fused before final classification.

This design is intended to capture both:

- high-level semantic information
- lower-level spatial detail relevant to visible spoilage patterns

## Project Structure

```text
uresnet/
├── config.py
├── dataset.py
├── model.py
├── engine.py
├── train.py
└── README.md
```

### `config.py`

Stores training configuration and constants such as:

- dataset path
- image size
- batch size
- number of epochs
- learning rate
- checkpoint path
- device selection (`cpu` or `cuda`)

The dataset path is loaded from a `.env` file so local file paths do not need to be committed to the repository.

### `dataset.py`

Contains dataset and preprocessing utilities:

- custom dataset wrapper for `(path, label)` samples
- dataset mean/std computation
- training and evaluation transforms

### `model.py`

Defines the U-ResNet architecture:

- `ResidualBlock`
- `DoubleConv`
- `UBranch`
- `ResBranch`
- `UResNetClassifier`

### `engine.py`

Contains training and evaluation logic, including:

- one-epoch training
- one-epoch evaluation
- performance metric calculation

### `train.py`

Main entry point for the training pipeline. It:

1. loads the dataset
2. creates train/validation/test splits
3. applies preprocessing
4. builds dataloaders
5. trains the model
6. saves the best checkpoint
7. evaluates the best model on the test set

## Model Architecture

The model consists of two parallel branches.

1. **ResNet branch**
   The residual branch learns compact discriminative features using residual blocks with skip connections. These skip connections help stabilize training and make feature learning more effective.
2. **U-Net branch**
   The U-style branch learns hierarchical spatial features using stacked convolution and pooling operations. This helps the model preserve richer feature structure that may be useful for identifying visual spoilage cues such as:
   - texture changes
   - bruising
   - discoloration
   - mold patterns
3. **Feature fusion**
   The outputs of both branches are combined along the channel dimension, passed through a fusion layer, pooled globally, and then sent to a fully connected classification layer.

## Dataset Format

The training script expects the dataset to be arranged in `ImageFolder` format, where each class has its own folder:

```text
Fruit And Vegetable Diseases Dataset/
├── Apple__Healthy/
├── Apple__Rotten/
├── Banana__Healthy/
├── Banana__Rotten/
└── ...
```

Each subfolder name is treated as a separate class label.


## Preprocessing

The preprocessing pipeline is intentionally lightweight and task-appropriate. This decision was made in order to provide a far comparison with other models implementations.

### Training Transforms

Training images use the following transformations:

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ToTensor`
- `Normalize`

This introduces mild variation while preserving the visual cues needed for freshness classification.

### Evaluation Transforms

Validation and test images use deterministic transforms:

- `Resize`
- `CenterCrop`
- `ToTensor`
- `Normalize`

This ensures that model evaluation is stable and repeatable.

### Normalization

The script can compute dataset-specific mean and standard deviation values from the training set. This improves consistency during optimization and avoids relying entirely on generic ImageNet statistics.

## Data Splitting

The training pipeline uses a stratified split so that class proportions are preserved across:

- training set
- validation set
- test set

Default proportions:

- 70% training
- 15% validation
- 15% testing

This helps ensure fairer evaluation, especially for smaller classes.

## Training Procedure

The model is trained using:

- `CrossEntropyLoss`
- `AdamW` optimizer
- automatic mixed precision with CUDA when available

During training, the script tracks:

- training loss
- training accuracy
- training macro-F1
- validation loss
- validation accuracy
- validation macro-F1

The best-performing model is selected based on validation F1 score and saved as a checkpoint.

## Evaluation Metrics

The implementation focuses on classification quality using:

- loss
- accuracy
- macro-F1

### Why Macro-F1 Matters

Macro-F1 is especially important here because the dataset contains multiple classes and some classes may be more difficult or less frequent than others. Unlike accuracy, macro-F1 gives equal importance to each class and provides a better view of how balanced the model's performance really is.

## Environment Setup

1. Install dependencies:

```bash
pip install torch torchvision scikit-learn pillow tqdm python-dotenv numpy
```

2. Create a `.env` file. The dataset path should be stored in a local `.env` file:

```env
DATASET_ROOT=C:\Your_Path
```

Example:

```env
DATASET_ROOT=C:\Users\yourname\path\to\Fruit And Vegetable Diseases Dataset
```

3. Add `.env` to `.gitignore` to prevent local machine-specific paths from being committed.


## Running the Code

Run the training script from the module directory:

```bash
python train.py
```

The script will:

1. load the dataset from `DATASET_ROOT`
2. split the data into train/validation/test sets
3. compute normalization statistics if enabled
4. train the U-ResNet model
5. save the best checkpoint
6. evaluate the best checkpoint on the test set

## Outputs

During training, the script prints epoch-level performance to the terminal.

Typical outputs include:

- training F1
- validation F1
- epoch timing
- final test F1

The best model weights are saved to the configured checkpoint path.


## Design Rationale

This implementation was written to be:

- **modular**: Different responsibilities are separated across configuration, preprocessing, model definition, training logic, and execution.
- **readable**: The code uses concise docstrings and clear function boundaries so collaborators can understand and extend it easily.
- **reproducible**: The training process uses fixed seeds, deterministic dataset splitting, and explicit preprocessing steps.
- **collaborative**: The structure is suitable for inclusion in a shared GitHub repository where multiple team members may be working on different model approaches.


## Current Limitations

This implementation is a strong baseline, but there are several natural extensions:

* Class imbalance impacts performance
    The gap between accuracy and macro-F1 shows the model performs better on larger classes, with weaker results on low-support classes like Pomegranate__Rotten and Guava__Rotten.
* Uneven class performance
    Some classes are near-perfect, while others are significantly weaker, reducing reliability across all categories.
* Rotten classes are harder to classify
    Rotten samples show more variability, leading to consistently lower performance compared to their healthy counterparts.
* Low recall on minority classes
    The model often misses harder positive examples, indicating it struggles with less obvious cases.
* No explicit imbalance handling
    The current setup uses standard cross-entropy without class weighting, sampling strategies, or focal loss.
* Limited error analysis
    While metrics highlight weaknesses, there is no deeper analysis of failure cases or label quality.

Overall, the model is a strong baseline, but future improvements should focus on handling class imbalance and improving performance on difficult classes rather than overall accuracy.
