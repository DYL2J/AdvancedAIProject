import cv2
import matplotlib.pyplot as plt
import numpy as np

from Modelling.RottenFresh.utils import load_image


def grabcut_mask(img, debug=False):

    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    # assume the obj is roughly centered
    rect = (10, 10, w - 20, h - 20)

    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8") * 255

    return mask


def apply_mask(img, mask, background="black"):
    """
    Apply binary mask to RGB image.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if background == "black":
        out = np.zeros_like(img)
    elif background == "white":
        out = np.ones_like(img) * 255
    else:
        out = img.copy()

    out[mask > 0] = img[mask > 0]
    return out


def crop_to_mask(img, mask, pad=10):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    return img[y1:y2, x1:x2]


def show_sample_segmentations(sample_df, n=4, title=None):
    if len(sample_df) == 0:
        print("No images to display.")
        return

    sample_df = sample_df.sample(min(n, len(sample_df)), random_state=42)

    fig, axes = plt.subplots(len(sample_df), 4, figsize=(16, 4 * len(sample_df)))
    if len(sample_df) == 1:
        axes = np.array([axes])

    for row_axes, (_, row) in zip(axes, sample_df.iterrows()):
        img = load_image(row["path"])
        mask = grabcut_mask(img)
        segmented = apply_mask(img, mask, background="black")
        cropped = crop_to_mask(img, mask)

        row_axes[0].imshow(img)
        row_axes[0].set_title("Original")
        row_axes[0].axis("off")

        row_axes[1].imshow(mask, cmap="gray")
        row_axes[1].set_title("Mask")
        row_axes[1].axis("off")

        row_axes[2].imshow(segmented)
        row_axes[2].set_title("Segmented")
        row_axes[2].axis("off")

        row_axes[3].imshow(cropped)
        row_axes[3].set_title("Cropped")
        row_axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def safe_percent(x):
    return float(np.clip(x * 100.0, 0, 100))


def load_image_resized(path, max_dim=256):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Remove isolated black/white pixel noise
    image_rgb = cv2.medianBlur(image_rgb, 3)

    h, w = image_rgb.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_rgb


def compute_shape_features(mask):
    mask_bin = (mask > 0).astype(np.uint8)

    area = mask_bin.sum()
    h, w = mask_bin.shape
    image_area = h * w

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "mask_area": 0,
            "area_ratio": 0.0,
            "bbox_area": 0,
            "extent": 0.0,
            "solidity": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
        }

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    bbox_area = bbox_w * bbox_h

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        perimeter = 0.0
        solidity = 0.0
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(cnt)
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    return {
        "mask_area": int(area),
        "area_ratio": area / image_area,
        "bbox_area": int(bbox_area),
        "extent": area / bbox_area if bbox_area > 0 else 0.0,
        "solidity": float(solidity),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
    }


def compute_color_features(img, mask):
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return {
            "mean_r": 0,
            "mean_g": 0,
            "mean_b": 0,
            "mean_h": 0,
            "mean_s": 0,
            "mean_v": 0,
            "std_h": 0,
            "std_s": 0,
            "std_v": 0,
            "dark_ratio": 0,
            "brown_ratio": 0,
        }

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    rgb_pixels = img[mask_bool]
    hsv_pixels = hsv[mask_bool]

    mean_r, mean_g, mean_b = rgb_pixels.mean(axis=0)
    mean_h, mean_s, mean_v = hsv_pixels.mean(axis=0)
    std_h, std_s, std_v = hsv_pixels.std(axis=0)

    # Dark/decayed proxy
    dark_ratio = np.mean(hsv_pixels[:, 2] < 80)

    # Broad brown-ish proxy in HSV (OpenCV H range is 0..179)
    h_vals = hsv_pixels[:, 0]
    s_vals = hsv_pixels[:, 1]
    v_vals = hsv_pixels[:, 2]
    brown_mask = (
        (h_vals >= 5)
        & (h_vals <= 25)
        & (s_vals >= 40)
        & (v_vals >= 20)
        & (v_vals <= 180)
    )
    brown_ratio = np.mean(brown_mask)

    return {
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "mean_h": float(mean_h),
        "mean_s": float(mean_s),
        "mean_v": float(mean_v),
        "std_h": float(std_h),
        "std_s": float(std_s),
        "std_v": float(std_v),
        "dark_ratio": float(dark_ratio),
        "brown_ratio": float(brown_ratio),
    }


def compute_texture_features(img, mask):
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return {
            "laplacian_var": 0.0,
            "gray_std": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_pixels = gray[mask_bool]

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap[mask_bool].var() if mask_bool.sum() > 0 else 0.0

    return {
        "laplacian_var": float(lap_var),
        "gray_std": float(gray_pixels.std()),
    }


def extract_features_for_image(path, max_dim=256):
    img = load_image_resized(path, max_dim=max_dim)
    mask = grabcut_mask(img)

    shape_feats = compute_shape_features(mask)
    color_feats = compute_color_features(img, mask)
    texture_feats = compute_texture_features(img, mask)

    return {**shape_feats, **color_feats, **texture_feats}


def plot_feature_distributions(features_df):
    cols_to_plot = [
        "mean_s",
        "dark_ratio",
        "brown_ratio",
        "laplacian_var",
        "area_ratio",
        "solidity",
    ]

    for col in cols_to_plot:
        plt.figure(figsize=(7, 4))
        for label in ["healthy", "rotten"]:
            subset = features_df.loc[
                features_df["quality_label"] == label, col
            ].dropna()
            plt.hist(subset, bins=40, alpha=0.5, label=label)
        plt.title(col)
        plt.legend()
        plt.show()
