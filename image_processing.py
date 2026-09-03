"""Module 1: the pipeline's single source of image loading and preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ImageData:
    """Canonical image representations shared by downstream modules."""

    path: Path
    image_id: str
    source_type: str
    original_bgr_uint8: np.ndarray  # H x W x 3, uint8; useful for OpenCV output
    original_rgb: np.ndarray       # H x W x 3, float32, range [0, 1]
    grayscale_uint8: np.ndarray    # H x W, uint8, range [0, 255]
    classifier_tensor: torch.Tensor  # 1 x 3 x H x W, float32, CPU
    original_height: int
    original_width: int
    coordinate_space: str = "original_image"

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "image_id": self.image_id,
            "source_type": self.source_type,
            "original_height": self.original_height,
            "original_width": self.original_width,
            "coordinate_space": self.coordinate_space,
            "rgb_dtype": str(self.original_rgb.dtype),
            "grayscale_dtype": str(self.grayscale_uint8.dtype),
            "classifier_tensor_shape": list(self.classifier_tensor.shape),
        }

    def classifier_tensor_on(self, device: str | torch.device) -> torch.Tensor:
        """Return the already-preprocessed classifier tensor on a target device."""
        return self.classifier_tensor.to(device)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Robustly map DICOM intensities to display/inference uint8."""
    image = np.asarray(array, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def _load_dicom(path: Path) -> np.ndarray:
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except ImportError as exc:
        raise ImportError("DICOM input requires: pip install pydicom") from exc

    dataset = pydicom.dcmread(str(path))
    pixels = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels
    return _normalize_to_uint8(pixels)


def load_image(path: str | Path) -> ImageData:
    """Load a raster or DICOM X-ray without changing its spatial dimensions."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image does not exist: {resolved}")

    if resolved.suffix.lower() == ".dcm":
        grayscale = _load_dicom(resolved)
        rgb_uint8 = np.repeat(grayscale[:, :, None], 3, axis=2)
        bgr = rgb_uint8.copy()
        source_type = "dicom"
    else:
        bgr = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unsupported or unreadable image: {resolved}")
        rgb_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        source_type = "raster"

    rgb = rgb_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    height, width = grayscale.shape

    return ImageData(
        path=resolved,
        image_id=resolved.stem,
        source_type=source_type,
        original_bgr_uint8=bgr,
        original_rgb=rgb,
        grayscale_uint8=grayscale,
        classifier_tensor=tensor.float(),
        original_height=height,
        original_width=width,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    loaded = load_image(args.image)
    for key, value in loaded.metadata().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
