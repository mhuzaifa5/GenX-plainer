"""Module 5: combine anatomical segmentation and predicted bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from classifier import ClassificationResult
from image_processing import ImageData
from segmentation import SegmentationResult
from xai import AttributionResult


# Distinct BGR colors for the 12 lung zones; heart is magenta.
REGION_COLORS = (
    (180, 119, 31), (232, 199, 174), (14, 127, 255),
    (150, 187, 255), (44, 160, 44), (138, 223, 152),
    (40, 39, 214), (134, 152, 255), (189, 103, 148),
    (213, 176, 197), (75, 86, 140), (148, 156, 196),
)


@dataclass(frozen=True)
class OverlayResult:
    image_bgr: np.ndarray
    box_count: int
    region_count: int
    segmentation_status: str
    attribution_method: str
    coordinate_space: str = "original_image"


def _display_name(region_name: str) -> str:
    return region_name.replace("_lung_", " ").replace("_", "/")


def create_overlay(
    image: ImageData,
    prediction: ClassificationResult,
    segmentation: SegmentationResult,
    attribution: AttributionResult,
    segmentation_alpha: float = 0.32,
    attribution_alpha: float = 0.50,
) -> OverlayResult:
    """Return a BGR overlay in the original image coordinate system."""
    if not 0 <= segmentation_alpha <= 1:
        raise ValueError("segmentation_alpha must be between 0 and 1.")
    if not 0 <= attribution_alpha <= 1:
        raise ValueError("attribution_alpha must be between 0 and 1.")

    expected_shape = (image.original_height, image.original_width)
    if segmentation.class_map.shape != expected_shape:
        raise ValueError(
            f"Segmentation shape {segmentation.class_map.shape} does not match "
            f"image shape {expected_shape}."
        )
    if prediction.coordinate_space != image.coordinate_space:
        raise ValueError("Prediction and image coordinate spaces do not match.")
    if segmentation.coordinate_space != image.coordinate_space:
        raise ValueError("Segmentation and image coordinate spaces do not match.")
    if attribution.coordinate_space != image.coordinate_space:
        raise ValueError("Attribution and image coordinate spaces do not match.")
    if attribution.signed_map.shape != expected_shape:
        raise ValueError("Attribution and image shapes do not match.")

    overlay = image.original_bgr_uint8.copy()

    # Diverging attribution layer: blue=negative, red=positive. Opacity is
    # proportional to magnitude, so near-zero attribution leaves the X-ray clear.
    signed = np.nan_to_num(attribution.signed_map, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(np.abs(signed), 98))
    if scale > 0:
        normalized = np.clip(signed / scale, -1.0, 1.0)
        heatmap = np.zeros_like(overlay)
        heatmap[normalized < 0, 0] = (-normalized[normalized < 0] * 255).astype(np.uint8)
        heatmap[normalized > 0, 2] = (normalized[normalized > 0] * 255).astype(np.uint8)
        opacity = (np.abs(normalized) * attribution_alpha)[..., None]
        overlay = np.clip(
            overlay.astype(np.float32) * (1.0 - opacity)
            + heatmap.astype(np.float32) * opacity,
            0,
            255,
        ).astype(np.uint8)
    color_canvas = np.zeros_like(overlay)
    lung_region_names = [name for name in segmentation.regions if name != "heart"]

    for index, name in enumerate(lung_region_names):
        selected = np.asarray(segmentation.regions[name]) > 0
        color_canvas[selected] = REGION_COLORS[index % len(REGION_COLORS)]

    heart_selected = segmentation.heart_mask > 0
    color_canvas[heart_selected] = (255, 0, 255)
    anatomy_selected = (segmentation.lung_mask > 0) | heart_selected
    blended = cv2.addWeighted(
        overlay, 1.0 - segmentation_alpha, color_canvas, segmentation_alpha, 0
    )
    overlay[anatomy_selected] = blended[anatomy_selected]

    lung_contours, _ = cv2.findContours(
        segmentation.lung_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, lung_contours, -1, (0, 255, 0), 2)
    heart_contours, _ = cv2.findContours(
        segmentation.heart_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, heart_contours, -1, (255, 0, 255), 2)

    # Label each lung zone at its centroid.
    for name in lung_region_names:
        rows, columns = np.where(np.asarray(segmentation.regions[name]) > 0)
        if columns.size:
            cv2.putText(
                overlay,
                _display_name(name),
                (max(0, int(columns.mean()) - 35), int(rows.mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Bounding boxes are yellow so they remain distinct from anatomical outlines.
    for detection in prediction.detections:
        x1, y1, x2, y2 = detection.box
        x1 = int(np.clip(x1, 0, image.original_width - 1))
        x2 = int(np.clip(x2, 0, image.original_width - 1))
        y1 = int(np.clip(y1, 0, image.original_height - 1))
        y2 = int(np.clip(y2, 0, image.original_height - 1))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
        label = f"pneumonia {detection.score:.3f}"
        label_y = max(18, y1 - 7)
        cv2.putText(
            overlay, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )

    return OverlayResult(
        image_bgr=overlay,
        box_count=len(prediction.detections),
        region_count=len(lung_region_names),
        segmentation_status=segmentation.status,
        attribution_method=attribution.method,
    )


def save_overlay(result: OverlayResult, output_path: str | Path) -> Path:
    """Save an OverlayResult as an image and return its resolved path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result.image_bgr):
        raise OSError(f"Could not save overlay: {path}")
    return path
