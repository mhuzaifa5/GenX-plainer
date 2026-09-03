"""Module 4: ianpan chest X-ray segmentation and anatomical regions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from transformers import AutoModel

from image_processing import ImageData, load_image


MODEL_ID = "ianpan/chest-x-ray-basic"
MODEL_REVISION = "fec189cd739afa79402a78ce41b46b4d35d784ab"
SUPPORTED_TRANSFORMERS_VERSION = "4.47.0"
ZONE_FRACTIONS = (1 / 3, 2 / 3)


@dataclass(frozen=True)
class SegmentationResult:
    """Native-resolution masks and provenance returned to later modules."""

    class_map: np.ndarray
    lung_mask: np.ndarray
    heart_mask: np.ndarray
    image_left_lung: np.ndarray
    image_right_lung: np.ndarray
    anatomical_right_lung: np.ndarray
    anatomical_left_lung: np.ndarray
    regions: dict[str, np.ndarray]
    region_audit: list[dict[str, object]]
    status: str
    model_id: str
    model_revision: str
    device: str
    coordinate_space: str = "original_image"

    def metadata(self) -> dict[str, object]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "coordinate_space": self.coordinate_space,
            "mask_shape": list(self.class_map.shape),
            "region_names": list(self.regions),
            "region_audit": self.region_audit,
        }


def split_lungs_by_image_position(
    binary_mask: np.ndarray, min_area_fraction: float = 0.01
) -> dict[str, np.ndarray | str]:
    """Select the two dominant lung components and assign laterality."""
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    parts = [
        (index, int(stats[index, cv2.CC_STAT_AREA]), float(centroids[index, 0]))
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= min_area_fraction * mask.size
    ]
    empty = np.zeros_like(mask)
    if len(parts) < 2:
        return {
            "status": "not_two_reliable_components",
            "image_left": empty.copy(),
            "image_right": empty.copy(),
            "anatomical_right": empty.copy(),
            "anatomical_left": empty.copy(),
        }

    image_left, image_right = sorted(
        sorted(parts, key=lambda item: item[1], reverse=True)[:2],
        key=lambda item: item[2],
    )
    left_side = (labels == image_left[0]).astype(np.uint8)
    right_side = (labels == image_right[0]).astype(np.uint8)
    return {
        "status": "two_components",
        "image_left": left_side,
        "image_right": right_side,
        # Standard frontal display convention.
        "anatomical_right": left_side,
        "anatomical_left": right_side,
    }


def make_anatomical_regions(
    lungs: dict[str, np.ndarray | str],
    zone_fractions: tuple[float, float] = ZONE_FRACTIONS,
    include_medial_lateral: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Match the notebook's vertical thirds and per-lung medial/lateral split."""
    if lungs["status"] != "two_components":
        return {}, [{"region_name": "unavailable", "reason": lungs["status"]}]
    if not 0 < zone_fractions[0] < zone_fractions[1] < 1:
        raise ValueError("Zone fractions must satisfy 0 < upper < lower < 1.")

    regions: dict[str, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    for anatomy, key, image_side in (
        ("right_lung", "anatomical_right", "image-left"),
        ("left_lung", "anatomical_left", "image-right"),
    ):
        lung = np.asarray(lungs[key]).astype(bool)
        rows, columns = np.where(lung)
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        lung_mid_x = (x0 + x1) / 2
        cuts = [
            y0,
            round(y0 + (y1 - y0) * zone_fractions[0]),
            round(y0 + (y1 - y0) * zone_fractions[1]),
            y1,
        ]

        for zone, start, stop in zip(("upper", "middle", "lower"), cuts[:-1], cuts[1:]):
            vertical = lung.copy()
            vertical[:start] = False
            vertical[stop:] = False
            if include_medial_lateral:
                x_grid = np.indices(lung.shape)[1]
                medial = x_grid >= lung_mid_x if anatomy == "right_lung" else x_grid < lung_mid_x
                for subdivision in ("medial", "lateral"):
                    selected = vertical & (medial if subdivision == "medial" else ~medial)
                    regions[f"{anatomy}_{zone}_{subdivision}"] = selected.astype(np.uint8)
            else:
                regions[f"{anatomy}_{zone}"] = vertical.astype(np.uint8)

        audit.append(
            {
                "anatomical_lung": anatomy,
                "image_space_side": image_side,
                "native_y_min": y0,
                "native_y_max_exclusive": y1,
                "native_x_min": x0,
                "native_x_max_exclusive": x1,
                "lung_mid_x": lung_mid_x,
            }
        )
    return regions, audit


class ChestXraySegmenter:
    """Reusable pipeline component; instantiate once and call segment repeatedly."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        device: str | torch.device | None = None,
    ) -> None:
        if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"{MODEL_ID} requires transformers=={SUPPORTED_TRANSFORMERS_VERSION} "
                f"for this pipeline, but {transformers.__version__} is installed. "
                "Install the pinned dependencies from requirements-modules.txt."
            )
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
        ).eval().to(self.device)
        if not hasattr(self.model, "preprocess"):
            raise AttributeError("Segmentation checkpoint does not provide preprocess().")

    def segment(self, grayscale_uint8: np.ndarray) -> SegmentationResult:
        """Run segmentation and return every mask in input-image coordinates."""
        gray = np.asarray(grayscale_uint8)
        if gray.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image; received shape {gray.shape}.")
        if gray.dtype != np.uint8:
            raise TypeError("grayscale_uint8 must have dtype uint8 and values in [0, 255].")

        processed = np.asarray(self.model.preprocess(gray))
        tensor = torch.from_numpy(processed).float()[None, None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        if not isinstance(output, dict) or "mask" not in output:
            raise TypeError("Expected model output dictionary containing 'mask'.")

        class_map = output["mask"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        class_map = cv2.resize(
            class_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        lung_mask = np.isin(class_map, [1, 2]).astype(np.uint8)
        heart_mask = (class_map == 3).astype(np.uint8)
        lungs = split_lungs_by_image_position(lung_mask)
        regions, audit = make_anatomical_regions(lungs)
        regions["heart"] = heart_mask

        return SegmentationResult(
            class_map=class_map,
            lung_mask=lung_mask,
            heart_mask=heart_mask,
            image_left_lung=np.asarray(lungs["image_left"]),
            image_right_lung=np.asarray(lungs["image_right"]),
            anatomical_right_lung=np.asarray(lungs["anatomical_right"]),
            anatomical_left_lung=np.asarray(lungs["anatomical_left"]),
            regions=regions,
            region_audit=audit,
            status=str(lungs["status"]),
            model_id=self.model_id,
            model_revision=self.model_revision,
            device=str(self.device),
        )

    def segment_image_data(self, image: ImageData) -> SegmentationResult:
        """Convenience adapter for Module 1 output."""
        result = self.segment(image.grayscale_uint8)
        expected = (image.original_height, image.original_width)
        if result.class_map.shape != expected:
            raise RuntimeError(f"Segmentation shape {result.class_map.shape} != image shape {expected}.")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    image = load_image(args.image)
    result = ChestXraySegmenter().segment_image_data(image)
    print(result.metadata())


if __name__ == "__main__":
    main()
