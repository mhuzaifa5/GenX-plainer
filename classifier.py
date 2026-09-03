"""Module 2: Faster R-CNN pneumonia classification and box prediction.

Image loading and preprocessing intentionally live only in image_processing.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from image_processing import ImageData, load_image


NUM_CLASSES = 2  # background + pneumonia


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax; original coordinates
    score: float
    label_id: int = 1
    label: str = "pneumonia"


@dataclass(frozen=True)
class ClassificationResult:
    predicted_class: str
    detections: list[Detection]
    detection_threshold: float
    coordinate_space: str = "original_image"

    def as_dict(self) -> dict[str, object]:
        return {
            "predicted_class": self.predicted_class,
            "detections": [
                {
                    "box": list(item.box),
                    "score": item.score,
                    "label_id": item.label_id,
                    "label": item.label,
                }
                for item in self.detections
            ],
            "detection_threshold": self.detection_threshold,
            "coordinate_space": self.coordinate_space,
        }


def build_model(num_classes: int = NUM_CLASSES, min_size: int = 1024) -> torch.nn.Module:
    """Build the exact Faster R-CNN architecture used by the supplied classifier."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model


def load_trained_model(
    weights_path: str | Path,
    device: str | torch.device | None = None,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.nn.Module, torch.device]:
    """Construct the detector and load its trained checkpoint once."""
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(num_classes=num_classes)
    checkpoint = torch.load(Path(weights_path), map_location=target_device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model.to(target_device).eval(), target_device


class PneumoniaClassifier:
    """Reusable Module 2 component; it never reads or modifies image files."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device | None = None,
        detection_threshold: float = 0.5,
    ) -> None:
        self.model, self.device = load_trained_model(weights_path, device)
        self.detection_threshold = detection_threshold

    def predict(self, image: ImageData) -> ClassificationResult:
        tensor = image.classifier_tensor_on(self.device)
        with torch.inference_mode():
            output = self.model(tensor)[0]

        score_mask = output["scores"] >= self.detection_threshold
        boxes = output["boxes"][score_mask]
        scores = output["scores"][score_mask]
        labels = output["labels"][score_mask]

        detections = [
            Detection(
                box=tuple(np.rint(box.detach().cpu().numpy()).astype(np.int32).tolist()),
                score=float(score.detach().cpu()),
                label_id=int(label.detach().cpu()),
            )
            for box, score, label in zip(boxes, scores, labels)
        ]
        return ClassificationResult(
            predicted_class="pneumonia" if detections else "normal",
            detections=detections,
            detection_threshold=self.detection_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    image = load_image(args.image)
    result = PneumoniaClassifier(args.weights).predict(image)
    print(result.as_dict())


if __name__ == "__main__":
    main()
