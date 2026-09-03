"""Run the available GenXplainer evidence pipeline end to end.

Pipeline: image processing -> classifier -> XAI -> segmentation -> KB -> overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classifier import PneumoniaClassifier
from image_processing import load_image
from knowledge_retrieval import PneumoniaKnowledgeRetriever
from overlay import create_overlay, save_overlay
from prompt_packet import build_prompt_packet, save_prompt_packet
from segmentation import ChestXraySegmenter
from xai import PneumoniaAttributor


def run_pipeline(
    image_path: str | Path,
    weights_path: str | Path,
    output_root: str | Path,
    kb_path: str | Path,
    ig_steps: int = 20,
) -> dict[str, object]:
    """Run the available modules and persist their structured artifacts."""
    print("[1/7] Loading and preprocessing image...")
    image = load_image(image_path)

    print("[2/7] Loading classifier and predicting boxes...")
    classifier = PneumoniaClassifier(weights_path)
    prediction = classifier.predict(image)

    print("[3/7] Calculating Integrated Gradients attribution...")
    attributor = PneumoniaAttributor(classifier, n_steps=ig_steps)
    attribution = attributor.attribute(image, prediction)

    print("[4/7] Loading segmentation model and predicting anatomy...")
    segmenter = ChestXraySegmenter()
    segmentation = segmenter.segment_image_data(image)

    case_dir = Path(output_root).expanduser().resolve() / image.image_id
    case_dir.mkdir(parents=True, exist_ok=True)

    print("[5/7] Retrieving grounded knowledge...")
    knowledge = PneumoniaKnowledgeRetriever(kb_path).retrieve(
        prediction, attribution, segmentation
    )

    print("[6/7] Creating attribution, prediction, and segmentation overlay...")
    overlay = create_overlay(image, prediction, segmentation, attribution)
    overlay_path = save_overlay(
        overlay, case_dir / "attribution_prediction_segmentation_overlay.png"
    )

    print("[7/7] Building the frozen Qwen prompt packet...")
    prompt_packet = build_prompt_packet(
        image, prediction, attribution, segmentation, knowledge, overlay_path
    )
    prompt_artifacts = save_prompt_packet(prompt_packet, case_dir)

    arrays: dict[str, np.ndarray] = {
        "class_map": segmentation.class_map,
        "lung_mask": segmentation.lung_mask,
        "heart_mask": segmentation.heart_mask,
        "image_left_lung": segmentation.image_left_lung,
        "image_right_lung": segmentation.image_right_lung,
        "anatomical_right_lung": segmentation.anatomical_right_lung,
        "anatomical_left_lung": segmentation.anatomical_left_lung,
    }
    arrays.update({f"region_{name}": mask for name, mask in segmentation.regions.items()})
    arrays_path = case_dir / "segmentation_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)

    attribution_path = case_dir / "integrated_gradients_attribution.npz"
    np.savez_compressed(
        attribution_path,
        signed_map=attribution.signed_map,
        absolute_map=attribution.absolute_map,
        channel_attribution=attribution.channel_attribution,
    )

    record: dict[str, object] = {
        "image": image.metadata(),
        "prediction": prediction.as_dict(),
        "segmentation": segmentation.metadata(),
        "attribution": attribution.metadata(),
        "knowledge": knowledge.as_dict(),
        "prompt_packet": prompt_packet.metadata(),
        "overlay": {
            "box_count": overlay.box_count,
            "region_count": overlay.region_count,
            "segmentation_status": overlay.segmentation_status,
            "coordinate_space": overlay.coordinate_space,
            "attribution_method": overlay.attribution_method,
        },
        "artifacts": {
            "segmentation_arrays": str(arrays_path),
            "integrated_gradients_attribution": str(attribution_path),
            "attribution_prediction_segmentation_overlay": str(overlay_path),
            "prompt_packet": str(prompt_artifacts["packet"]),
        },
    }
    record_path = case_dir / "pipeline_results.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("Pipeline completed.")
    print(f"Prediction: {prediction.predicted_class}")
    print(f"Detections: {len(prediction.detections)}")
    print(f"Segmentation status: {segmentation.status}")
    print(f"Results: {record_path}")
    print(f"Masks:   {arrays_path}")
    print(f"XAI:     {attribution_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Prompt:  {prompt_artifacts['prompt']}")
    return record


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb",
        type=Path,
        default=module_dir / "pneumonia_kb.csv",
        help="Local pneumonia knowledge-base CSV",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=module_dir / "chest_xray.png",
        help="Input image (default: chest_xray.png beside main.py)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=module_dir / "bestepoch_fasterrcnn_resnet50_fpn.pth",
        help="Faster R-CNN checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=module_dir / "outputs",
        help="Directory for pipeline artifacts",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=20,
        help="Integrated Gradients steps (default: 20; use 2 only for a quick test)",
    )
    args = parser.parse_args()
    run_pipeline(args.image, args.weights, args.output_dir, args.kb, args.ig_steps)


if __name__ == "__main__":
    main()
