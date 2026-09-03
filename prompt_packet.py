"""Build the final grounded evidence packet for the remote Qwen server."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from classifier import ClassificationResult
from image_processing import ImageData
from knowledge_retrieval import KnowledgeRetrievalResult
from segmentation import SegmentationResult
from xai import AttributionResult


DIAGNOSTIC_QUESTION = (
    "Analyze this chest X-ray for evidence relevant to pneumonia. Describe the "
    "radiographic evidence you can support, where it is located anatomically, and "
    "whether the available evidence supports pneumonia. Distinguish observations "
    "from interpretations and do not claim findings unsupported by the supplied evidence."
)

RESPONSE_TEMPLATE = {
    "observed_findings": [],
    "anatomical_locations": [],
    "pneumonia_assessment": "",
    "supporting_evidence": [],
    "uncertainties": [],
    "unsupported_or_conflicting_evidence": [],
    "confidence": 0.0,
}


@dataclass(frozen=True)
class PromptPacket:
    case_id: str
    prompt_text: str
    evidence: dict[str, object]
    response_template: dict[str, object]
    attachment_paths: list[Path]

    def metadata(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "attachment_count": len(self.attachment_paths),
            "attachments": [str(path) for path in self.attachment_paths],
            "response_format": "json",
        }


def summarize_region_attribution(
    attribution: AttributionResult,
    segmentation: SegmentationResult,
    top_n: int = 8,
) -> list[dict[str, object]]:
    """Rank regions by mean absolute attribution, normalized for region size."""
    absolute = attribution.absolute_map
    signed = attribution.signed_map
    total_absolute = float(absolute.sum())
    if total_absolute <= 0:
        return []
    records: list[dict[str, object]] = []

    for name, region_mask in segmentation.regions.items():
        selected = np.asarray(region_mask) > 0
        pixel_count = int(selected.sum())
        if pixel_count == 0:
            continue
        absolute_mass = float(absolute[selected].sum())
        signed_sum = float(signed[selected].sum())
        positive_sum = float(np.maximum(signed[selected], 0).sum())
        negative_sum = float(np.minimum(signed[selected], 0).sum())
        records.append(
            {
                "region_name": name,
                "region_type": "heart" if name == "heart" else "lung_zone",
                "attribution_mass": absolute_mass,
                "mean_absolute_attribution": absolute_mass / pixel_count,
                "attribution_fraction_total": (
                    absolute_mass / total_absolute if total_absolute else 0.0
                ),
                "signed_attribution_sum": signed_sum,
                "signed_mean_attribution": signed_sum / pixel_count,
                "positive_signed_attribution_sum": positive_sum,
                "negative_signed_attribution_sum": negative_sum,
                "pixel_count": pixel_count,
                "segmentation_status": segmentation.status,
            }
        )

    records.sort(
        key=lambda record: (
            -float(record["mean_absolute_attribution"]),
            -float(record["attribution_fraction_total"]),
        )
    )
    return records[:top_n]


def build_prompt_packet(
    image: ImageData,
    prediction: ClassificationResult,
    attribution: AttributionResult,
    segmentation: SegmentationResult,
    knowledge: KnowledgeRetrievalResult,
    overlay_path: str | Path,
) -> PromptPacket:
    """Create a frozen prompt from upstream evidence without calling an LLM."""
    resolved_overlay = Path(overlay_path).expanduser().resolve()
    if not resolved_overlay.is_file():
        raise FileNotFoundError(f"Overlay does not exist: {resolved_overlay}")

    region_records = summarize_region_attribution(attribution, segmentation, top_n=8)
    detections = [
        {
            "label": item.label,
            "score": item.score,
            "box_xyxy": list(item.box),
            "coordinate_space": prediction.coordinate_space,
        }
        for item in prediction.detections
    ]
    evidence: dict[str, object] = {
        "image": image.metadata(),
        "classifier": {
            "predicted_class": prediction.predicted_class,
            "detection_threshold": prediction.detection_threshold,
            "qualifying_detections": detections,
        },
        "attribution": {
            **attribution.metadata(),
            "regional_summary": region_records,
        },
        "segmentation": segmentation.metadata(),
        "knowledge_base": knowledge.as_dict(),
    }

    # Compact projection for Qwen. Full precision and provenance stay in evidence.json.
    compact_detections = [
        {
            "label": item["label"],
            "score": round(float(item["score"]), 3),
            "box_xyxy": item["box_xyxy"],
        }
        for item in detections
    ]
    compact_classifier = {
        "predicted_class": prediction.predicted_class,
        "detection_threshold": round(prediction.detection_threshold, 3),
        "qualifying_detections": compact_detections,
    }
    compact_regions = [
        {
            "region": record["region_name"],
            "mean_absolute_attribution": round(
                float(record["mean_absolute_attribution"]), 6
            ),
            "attribution_fraction_total": round(
                float(record["attribution_fraction_total"]), 4
            ),
            "signed_mean_attribution": round(
                float(record["signed_mean_attribution"]), 6
            ),
        }
        for record in region_records[:3]
    ]
    compact_segmentation = {
        "status": segmentation.status,
        "coordinate_space": segmentation.coordinate_space,
        "zones": "upper/middle/lower thirds, each split medial/lateral",
        "laterality": "image-left=anatomical right; image-right=anatomical left",
    }
    compact_knowledge = [
        {
            "finding": record.finding_label,
            "relation": record.relation,
            "source": record.source,
            "finding_uri": record.finding_rgo_uri,
            "mapping_status": record.mapping_status,
        }
        for record in knowledge.records[:3]
    ]

    if prediction.predicted_class == "pneumonia":
        attribution_guidance = (
            "Integrated Gradients indicates pixels and regions influential to the "
            "qualifying pneumonia detection and is not proof of pathology. Positive "
            "signed attribution supports the detection relative to the black baseline; "
            "negative attribution opposes it. Mean absolute attribution is normalized "
            "by region size and is the primary regional ranking metric."
        )
    else:
        attribution_guidance = (
            "No pneumonia detection met the classifier threshold, so there is no "
            "qualifying pneumonia target. Attribution arrays and regional attribution "
            "records are intentionally empty/zero; do not interpret sub-threshold candidates."
        )

    sections = [
        DIAGNOSTIC_QUESTION,
        "Return only valid JSON matching this exact shape:\n"
        + json.dumps(RESPONSE_TEMPLATE, indent=2),
        (
            "Confidence must be between 0 and 1. The detector output is auxiliary "
            "model evidence, not ground truth. Do not invent a global classifier "
            "score; use only the qualifying detection scores supplied below."
        ),
        "Classifier decision and qualifying boxes:\n"
        + json.dumps(compact_classifier, indent=2),
        attribution_guidance,
        "Quantitative attribution by anatomical region:\n"
        + json.dumps(compact_regions, indent=2),
        (
            "The anatomical zones are geometric subdivisions of predicted masks and "
            "may be imperfect. Image-left corresponds to anatomical right on a standard "
            "frontal display. Segmentation summary:\n"
            + json.dumps(compact_segmentation, indent=2)
        ),
        (
            "The following ontology records are general possible relationships, not "
            "patient-specific observations. Do not assert a KB concept unless the "
            "images and supplied case evidence support it:\n"
            + json.dumps(compact_knowledge, indent=2)
        ),
        (
            "Use only the original image and this frozen evidence packet. Clearly mark "
            "uncertainty, separate image observations from model-derived evidence, and "
            "place unsupported or conflicting suggestions in the designated JSON field."
        ),
    ]
    return PromptPacket(
        case_id=image.image_id,
        prompt_text="\n\n".join(sections),
        evidence=evidence,
        response_template=RESPONSE_TEMPLATE.copy(),
        attachment_paths=[image.path, resolved_overlay],
    )


def save_prompt_packet(packet: PromptPacket, output_dir: str | Path) -> dict[str, Path]:
    """Save the prompt, evidence, response template, and ordered attachments."""
    packet_dir = Path(output_dir).expanduser().resolve() / "prompt_packet"
    packet_dir.mkdir(parents=True, exist_ok=True)

    copied_attachments: list[str] = []
    for index, source in enumerate(packet.attachment_paths, start=1):
        destination = packet_dir / f"attachment_{index}_{source.name}"
        shutil.copy2(source, destination)
        copied_attachments.append(destination.name)

    prompt_path = packet_dir / "prompt.txt"
    evidence_path = packet_dir / "evidence.json"
    manifest_path = packet_dir / "attachments_manifest.json"
    template_path = packet_dir / "response_template.json"
    packet_path = packet_dir / "prompt_packet.json"

    prompt_path.write_text(packet.prompt_text, encoding="utf-8")
    evidence_path.write_text(json.dumps(packet.evidence, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"upload_in_this_order": copied_attachments}, indent=2),
        encoding="utf-8",
    )
    template_path.write_text(
        json.dumps(packet.response_template, indent=2), encoding="utf-8"
    )
    packet_path.write_text(
        json.dumps(
            {
                "case_id": packet.case_id,
                "prompt": packet.prompt_text,
                "attachments": copied_attachments,
                "response_format": "json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "directory": packet_dir,
        "prompt": prompt_path,
        "evidence": evidence_path,
        "manifest": manifest_path,
        "response_template": template_path,
        "packet": packet_path,
    }
