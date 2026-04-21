from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from feature_extract import (
    compute_color_features,
    compute_shape_features,
    compute_texture_features,
    grabcut_mask,
    load_image_resized,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASS_NAMES = ["fresh", "rotten"]


class ProductAnalyzer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        output_dir: str | Path = "analysis_outputs",
        class_names: list[str] | None = None,
        image_size: int = 224,
        shap_blur: str = "blur(32,32)",
        shap_max_evals: int = 300,
        shap_batch_size: int = 1,
        device: str | None = None,
    ) -> None:

        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.image_size = image_size
        self.shap_blur = shap_blur
        self.shap_max_evals = shap_max_evals
        self.shap_batch_size = shap_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """
        Loads the model and weights for freshness classification.
        Expects a checkpoint file containing the state dict of the model.
        Returns:
            A PyTorch nn.Module with loaded weights, set to evaluation mode.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path.resolve()}"
            )

        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.class_names))
        model = model.to(self.device)

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _preprocess_rgb_numpy(self, images_rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a batch of RGB images in numpy format for model input.
        Args:
            images_rgb: A numpy array of shape (N, H, W, 3) with RGB values in [0, 1] or [0, 255].
        Returns:
            A torch.Tensor of shape (N, 3, H, W) normalized for model input.
        """

        if images_rgb.dtype != np.float32:
            images_rgb = images_rgb.astype(np.float32)

        if images_rgb.max() > 1.0:
            images_rgb = images_rgb / 255.0

        tensor = torch.from_numpy(images_rgb).permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_from_rgb_numpy(self, images_rgb: np.ndarray) -> np.ndarray:
        x = self._preprocess_rgb_numpy(images_rgb)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def load_input_image(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path.resolve()}")

        with Image.open(image_path) as img:
            image = img.convert("RGBA").convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            return np.asarray(image).astype(np.float32) / 255.0

    def _save_freshness_result(
        self, rgb_image: np.ndarray, result: dict[str, Any]
    ) -> Path:
        out_path = self.output_dir / "freshness_prediction.png"

        plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(rgb_image, 0, 1))

        pred_label = result["freshness_prediction"]
        confidence = result["freshness_confidence"]
        fresh_prob = result["class_probabilities"].get("fresh", 0.0)
        rotten_prob = result["class_probabilities"].get("rotten", 0.0)

        ax.set_title(
            f"Prediction: {pred_label.upper()}\n"
            f"Confidence: {confidence:.2%} | Fresh: {fresh_prob:.2%} | Rotten: {rotten_prob:.2%}",
            fontsize=11,
        )

        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def run_freshness_evaluation(self, rgb_image: np.ndarray) -> dict[str, Any]:
        """
        Evalutates the freshness of the produce using the loaded model.
        Args:
            rgb_image: A numpy array of shape (H, W, 3) with RGB values in [0, 1].
        Returns:
            A dictionary containing the predicted freshness label, confidence score, predicted class index, and probabilities for each class.
        """

        probs = self.predict_from_rgb_numpy(np.expand_dims(rgb_image, axis=0))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]

        return {
            "freshness_prediction": pred_label,
            "freshness_confidence": float(probs[pred_idx]),
            "predicted_index": pred_idx,
            "class_probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs)
            },
        }

    def _reduce_shap_values(self, shap_values: shap.Explanation) -> np.ndarray:
        """
        Reduces the SHAP values to a 3D array.
        Args:
            shap_values: A SHAP Explanation object containing the SHAP values for the input image.
        Returns:
            A numpy array of shape (H, W, 3) representing the SHAP values for each RGB channel.
        """

        values = np.array(shap_values.values)

        if values.ndim == 4:
            sample_values = values[0]
        elif values.ndim == 5 and values.shape[-1] == 1:
            sample_values = values[0, ..., 0]
        elif values.ndim == 5:
            sample_values = values[0, ..., 0]
        else:
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

        if sample_values.ndim != 3:
            raise ValueError(f"Unexpected reduced SHAP shape: {sample_values.shape}")

        return sample_values

    def _make_defect_overlay(
        self,
        rgb_image_uint8: np.ndarray,
        shap_map: np.ndarray,
        object_mask: np.ndarray,
        threshold_percentile: float = 85.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates an overlay highlighting potential defect regions based on SHAP values.
        Args:
            rgb_image_uint8: The original input image as a uint8 numpy array of shape (H, W, 3).
            shap_map: A numpy array of shape (H, W, 3) containing the SHAP values for each pixel and channel.
            object_mask: A binary mask of shape (H, W) indicating the segmented object region.
            threshold_percentile: The percentile threshold to determine which SHAP values indicate potential defects.
        Returns:
            A tuple containing:
                - An RGB image with defect regions highlighted (numpy array of shape (H, W, 3)).
                - A binary mask of shape (H, W) where defect regions are marked with 255 and non-defect regions are 0.
        """

        importance = np.abs(shap_map).mean(axis=2)

        if object_mask is not None and np.any(object_mask > 0):
            masked_importance = importance[object_mask > 0]
        else:
            masked_importance = importance.reshape(-1)

        if masked_importance.size == 0:
            threshold = np.percentile(importance, threshold_percentile)
        else:
            threshold = np.percentile(masked_importance, threshold_percentile)

        defect_mask = (importance >= threshold).astype(np.uint8)
        if object_mask is not None:
            defect_mask = defect_mask * (object_mask > 0).astype(np.uint8)

        overlay = rgb_image_uint8.copy()
        red_layer = overlay.copy()
        red_layer[:, :, 0] = 255
        red_layer[:, :, 1] = 40
        red_layer[:, :, 2] = 40

        alpha = defect_mask[..., None].astype(np.float32) * 0.45
        overlay = (
            overlay.astype(np.float32) * (1 - alpha)
            + red_layer.astype(np.float32) * alpha
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay, defect_mask * 255

    def run_defect_detection(self, rgb_image: np.ndarray) -> dict[str, Any]:
        """
        Detects potential defect regions in the produce using SHAP explanations of the freshness prediction.
        Args:
            rgb_image: A numpy array of shape (H, W, 3) with RGB values in [0, 1].
        Returns:
            A dictionary containing paths to the SHAP explanation plot, defect highlight overlay, and defect mask, as well as summary statistics about the SHAP values and detected defects.
        """

        masker = shap.maskers.Image(self.shap_blur, rgb_image.shape)
        explainer = shap.Explainer(
            lambda x: self.predict_from_rgb_numpy(x),
            masker,
            output_names=self.class_names,
        )

        shap_values = explainer(
            np.expand_dims(rgb_image, axis=0),
            max_evals=self.shap_max_evals,
            batch_size=self.shap_batch_size,
            outputs=shap.Explanation.argsort.flip[:1],
        )

        rgb_uint8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
        object_mask = grabcut_mask(rgb_uint8)
        shap_map = self._reduce_shap_values(shap_values)
        overlay, defect_mask = self._make_defect_overlay(
            rgb_uint8, shap_map, object_mask
        )

        shap_plot_path = self.output_dir / "shap_explanation.png"
        overlay_path = self.output_dir / "defect_highlight_overlay.png"
        defect_mask_path = self.output_dir / "defect_mask.png"

        plt.close("all")
        shap.image_plot(shap_values, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        fig.savefig(shap_plot_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(defect_mask_path), defect_mask)

        mean_abs_shap = float(np.abs(shap_map).mean())
        max_abs_shap = float(np.abs(shap_map).max())
        defect_ratio = float((defect_mask > 0).mean())

        return {
            "shap_plot_path": str(shap_plot_path),
            "defect_overlay_path": str(overlay_path),
            "defect_mask_path": str(defect_mask_path),
            "mean_abs_shap": mean_abs_shap,
            "max_abs_shap": max_abs_shap,
            "defect_area_ratio": defect_ratio,
        }

    def run_quality_inspection(self, image_path: str | Path) -> dict[str, Any]:
        """
        Runs a rule-based quality inspection based on extracted features from the image.
        Args:
            image_path: The path to the input image.
        Returns:
            A dictionary containing the extracted features, quality assessment results, and paths to generated reports and segmented images.
        """

        image_path = Path(image_path)
        img = load_image_resized(image_path, max_dim=256)
        mask = grabcut_mask(img)

        shape_features = compute_shape_features(mask)
        color_features = compute_color_features(img, mask)
        texture_features = compute_texture_features(img, mask)

        features = {**shape_features, **color_features, **texture_features}
        quality = self._grade_quality(features)

        feature_report_path = self.output_dir / "quality_features.json"
        with feature_report_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": features,
                    "quality_assessment": quality,
                },
                f,
                indent=2,
            )

        segmented = np.zeros_like(img)
        segmented[mask > 0] = img[mask > 0]
        segmented_path = self.output_dir / "segmented_product.png"
        cv2.imwrite(str(segmented_path), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

        return {
            "features": features,
            "quality_assessment": quality,
            "feature_report_path": str(feature_report_path),
            "segmented_product_path": str(segmented_path),
        }

    def _grade_quality(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Grades the quality of the produce based on extracted features using a rule-based system.
        Args:
            features: A dictionary of extracted features including color, shape, and texture metrics.
        Returns:
            A dictionary containing the final quality score, grade, label, and detailed deductions based on the feature values.
        """

        score = 100.0
        deductions: list[dict[str, Any]] = []

        def deduct(
            condition: bool, points: float, reason: str, value: float | None = None
        ) -> None:
            nonlocal score
            if condition:
                score -= points
                entry: dict[str, Any] = {"reason": reason, "points": points}
                if value is not None:
                    entry["measured_value"] = float(value)
                deductions.append(entry)

        dark_ratio = float(features.get("dark_ratio", 0.0))
        brown_ratio = float(features.get("brown_ratio", 0.0))
        solidity = float(features.get("solidity", 0.0))
        circularity = float(features.get("circularity", 0.0))
        gray_std = float(features.get("gray_std", 0.0))
        laplacian_var = float(features.get("laplacian_var", 0.0))
        area_ratio = float(features.get("area_ratio", 0.0))
        extent = float(features.get("extent", 0.0))

        # Deduction rules based on feature threshholds, change descriptions and values if needed
        deduct(
            dark_ratio > 0.20,
            18,
            "High dark regions ratio detected, indicating possible bruising or decay.",
            dark_ratio,
        )
        deduct(
            dark_ratio > 0.35,
            12,
            "Very high dark region ratio detected, indicating high levels of rot.",
            dark_ratio,
        )
        deduct(
            brown_ratio > 0.15,
            15,
            "High brown region ratio, indicating browning or rot.",
            brown_ratio,
        )
        deduct(brown_ratio > 0.30, 10, "Severe browning detected.", brown_ratio)
        deduct(
            solidity < 0.90,
            10,
            "Low solidity detected, indicating prescence of dents, irregularity, or deformation.",
            solidity,
        )
        deduct(
            circularity < 0.55,
            4,
            "Low circularity detected, suggesting misshapen produce.",
            circularity,
        )
        deduct(
            gray_std > 55,
            8,
            "High grayscale variation detected, suggesting uneven surface quality.",
            gray_std,
        )
        deduct(
            laplacian_var > 2500, 7, "Rough or noisy texture detected.", laplacian_var
        )
        deduct(
            area_ratio < 0.20,
            6,
            "Object occupies too little of the frame for robust inspection.",
            area_ratio,
        )
        deduct(
            extent < 0.55, 6, "Fruit coverage inside the bounding box is low.", extent
        )

        score = float(np.clip(score, 0, 100))

        if score >= 95:
            grade = "A"
            label = "Excellent"
        elif score >= 90:
            grade = "B"
            label = "Good"
        elif score >= 80:
            grade = "C"
            label = "Fair"
        elif score >= 75:
            grade = "D"
            label = "Poor"
        elif score >= 60:
            grade = "E"
            label = "Atrocious"
        else:
            grade = "F"
            label = "Reject"

        return {
            "quality_score": score,
            "quality_grade": grade,
            "quality_label": label,
            "deductions": deductions,
            "interpretation": (
                "Rule-based manual feature assessment using color, shape, and texture cues."
            ),
        }

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        """
        Runs the full analysis pipeline on the input image, including freshness evaluation, defect detection, and quality inspection.
        Args:
            image_path: The path to the input image to be analyzed.
        Returns:
            A comprehensive dictionary containing results from all analysis stages, paths to generated outputs, and summary statistics.
        """

        image_path = Path(image_path)
        image_stem = image_path.stem
        self.output_dir = self.output_dir / image_stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rgb_image = self.load_input_image(image_path)

        freshness_result = self.run_freshness_evaluation(rgb_image)
        freshness_plot_path = self._save_freshness_result(rgb_image, freshness_result)
        defect_result = self.run_defect_detection(rgb_image)
        quality_result = self.run_quality_inspection(image_path)

        result = {
            "input_image": str(image_path),
            "device": self.device,
            "freshness_evaluation": {
                **freshness_result,
                "freshness_plot_path": str(freshness_plot_path),
            },
            "defect_detection": defect_result,
            "quality_inspection": quality_result,
        }

        summary_path = self.output_dir / "analysis_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["summary_path"] = str(summary_path)
        return result


def analyze_product(
    image_path: str | Path,
    checkpoint_path: str | Path = "best_resnet18_utils.pth",
    output_dir: str | Path = "analysis_outputs",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyzes a produce image for freshness, defects, and quality using a combination of model predictions and rule-based feature assessments.
    Args:
        image_path: The path to the input image to be analyzed.
        checkpoint_path: The path to the trained model checkpoint for freshness classification.
        output_dir: The directory where generated outputs and reports will be saved.
        class_names: Optional list of class names corresponding to the model's output indices. Defaults to ["fresh", "rotten"] if not provided.
    Returns:
        A comprehensive dictionary containing results from all analysis stages, paths to generated outputs, and summary statistics.
    """
    analyzer = ProductAnalyzer(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        class_names=class_names,
    )
    return analyzer.analyze(image_path)


def build_argparser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for command-line execution of the product analysis.
    Returns:
        An argparse.ArgumentParser object with defined arguments for image path, checkpoint path, and output directory.
    """
    parser = argparse.ArgumentParser(
        description="Analyze a produce image for freshness, defects, and quality."
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_resnet18_utils.pth",
        help="Path to trained ResNet18 checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs",
        help="Directory for generated outputs",
    )

    return parser


def main() -> None:
    """
    Main function for command-line execution. Parses arguments and runs the product analysis, printing the results.
    """

    parser = build_argparser()
    args = parser.parse_args()

    results = analyze_product(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print(results)

    # print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
