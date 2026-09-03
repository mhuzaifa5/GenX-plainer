"""Module 3: Integrated Gradients attribution for the pneumonia detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from classifier import ClassificationResult, PneumoniaClassifier
from image_processing import ImageData


@dataclass(frozen=True)
class AttributionResult:
    signed_map: np.ndarray       # H x W float32; direction of influence
    absolute_map: np.ndarray     # H x W float32; magnitude of influence
    channel_attribution: np.ndarray  # 3 x H x W float32
    convergence_delta: float
    target: str
    target_score: float
    method: str = "integrated_gradients"
    coordinate_space: str = "original_image"

    def metadata(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "target_score": self.target_score,
            "convergence_delta": self.convergence_delta,
            "map_shape": list(self.signed_map.shape),
            "coordinate_space": self.coordinate_space,
        }


class FasterRCNNPneumoniaWrapper(nn.Module):
    """Expose one differentiable maximum pneumonia score per input image."""

    def __init__(self, model: nn.Module, detection_threshold: float) -> None:
        super().__init__()
        self.model = model
        self.detection_threshold = detection_threshold

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model([image for image in batch])
        scores: list[torch.Tensor] = []
        for index, output in enumerate(outputs):
            pneumonia = (output["labels"] == 1) & (
                output["scores"] >= self.detection_threshold
            )
            if pneumonia.any():
                scores.append(output["scores"][pneumonia].max())
            else:
                # Preserve the graph if the detector emits no pneumonia candidate.
                scores.append(batch[index].sum() * 0.0)
        return torch.stack(scores)


class PneumoniaAttributor:
    """Reusable Integrated Gradients component sharing Module 2's model."""

    def __init__(
        self,
        classifier: PneumoniaClassifier,
        n_steps: int = 20,
        internal_batch_size: int = 1,
    ) -> None:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        self.classifier = classifier
        self.wrapper = FasterRCNNPneumoniaWrapper(
            classifier.model,
            detection_threshold=classifier.detection_threshold,
        ).to(classifier.device).eval()
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def attribute(
        self,
        image: ImageData,
        prediction: ClassificationResult,
    ) -> AttributionResult:
        """Attribute only the qualifying detection behind Module 2's decision."""
        tensor = image.classifier_tensor_on(self.classifier.device).detach()

        if prediction.predicted_class == "normal":
            channels = np.zeros(tuple(tensor.shape[1:]), dtype=np.float32)
            spatial = np.zeros(
                (image.original_height, image.original_width), dtype=np.float32
            )
            return AttributionResult(
                signed_map=spatial.copy(),
                absolute_map=spatial.copy(),
                channel_attribution=channels,
                convergence_delta=0.0,
                target="pneumonia",
                target_score=0.0,
            )

        if prediction.predicted_class != "pneumonia":
            raise ValueError(
                f"Unsupported predicted_class: {prediction.predicted_class!r}."
            )

        qualifying = [
            detection
            for detection in prediction.detections
            if detection.label_id == 1
            and detection.score >= self.classifier.detection_threshold
        ]
        if not qualifying:
            raise RuntimeError(
                "Module 2 classified the image as pneumonia but returned no class-1 "
                "detection at or above its detection threshold."
            )
        decision_score = max(detection.score for detection in qualifying)

        tensor.requires_grad_(True)
        baseline = torch.zeros_like(tensor)

        with torch.no_grad():
            wrapper_score = float(self.wrapper(tensor).squeeze().cpu())
        if wrapper_score <= 0.0:
            raise RuntimeError(
                "Module 2 classified the image as pneumonia, but the XAI wrapper "
                "found no qualifying pneumonia detection."
            )
        if not np.isclose(wrapper_score, decision_score, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "Module 2 and the XAI wrapper selected different pneumonia scores: "
                f"{decision_score:.8f} versus {wrapper_score:.8f}."
            )

        integrated_gradients = IntegratedGradients(self.wrapper)
        attribution, delta = integrated_gradients.attribute(
            tensor,
            baselines=baseline,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=True,
        )
        channels = attribution[0].detach().cpu().numpy().astype(np.float32)
        signed_map = channels.mean(axis=0).astype(np.float32)
        absolute_map = np.abs(channels).mean(axis=0).astype(np.float32)

        expected = (image.original_height, image.original_width)
        if signed_map.shape != expected:
            raise RuntimeError(
                f"Attribution shape {signed_map.shape} does not match image {expected}."
            )

        self.wrapper.zero_grad(set_to_none=True)
        return AttributionResult(
            signed_map=signed_map,
            absolute_map=absolute_map,
            channel_attribution=channels,
            convergence_delta=float(delta.detach().mean().cpu()),
            target="pneumonia",
            target_score=decision_score,
        )
