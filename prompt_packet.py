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

    # Compact, typed projection for Qwen. Full precision stays in evidence.json.
    compact_detections = [
        {
            "detection_id": f"det_{index}",
            "label": item["label"],
            "score": round(float(item["score"]), 3),
            "box_xyxy": item["box_xyxy"],
            "coordinate_space": prediction.coordinate_space,
        }
        for index, item in enumerate(detections, start=1)
    ]
    compact_decision = {
        "evidence_type": "detector_output",
        "model": "Faster R-CNN",
        "predicted_class": prediction.predicted_class,
        "detection_threshold": round(prediction.detection_threshold, 3),
        "decision_rule": (
            "pneumonia if at least one class-1 detection score meets threshold; "
            "otherwise normal"
        ),
        "score_semantics": "model detection score; not a calibrated disease probability",
        "qualifying_detections": compact_detections,
    }
    compact_regions = [
        {
            "rank": rank,
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
        for rank, record in enumerate(region_records[:3], start=1)
    ]
    compact_segmentation = {
        "evidence_type": "predicted_anatomical_segmentation",
        "status": segmentation.status,
        "coordinate_space": segmentation.coordinate_space,
        "region_definition": "upper/middle/lower thirds, each split medial/lateral",
        "laterality": "image-left=anatomical right; image-right=anatomical left",
    }
    compact_knowledge = [
        {
            "record_id": f"kb_{index}",
            "finding": record.finding_label,
            "relation": record.relation,
            "source": record.source,
            "finding_uri": record.finding_rgo_uri,
            "mapping_status": record.mapping_status,
        }
        for index, record in enumerate(knowledge.records[:3], start=1)
    ]

    if prediction.predicted_class == "pneumonia":
        attribution_status = "available_for_qualifying_pneumonia_detection"
    else:
        attribution_status = "no_qualifying_target_zero_attribution"

    compact_attribution = {
        "evidence_type": "model_attribution",
        "method": attribution.method,
        "status": attribution_status,
        "target": attribution.target,
        "target_score": round(float(attribution.target_score), 3),
        "baseline": "black image",
        "ranking_metric": "mean_absolute_attribution",
        "ranking_semantics": "model influence normalized by region pixel count",
        "top_regions": compact_regions,
        "limitations": (
            "Attribution indicates influence on the selected model output, not a semantic "
            "finding, pathology location, severity, diagnosis, or probability."
        ),
    }

    compact_relationships = {
        "detector_box_intersects_anatomical_regions": knowledge.box_overlap_regions,
        "intersection_definition": (
            "at least one qualifying detection-box pixel intersects the predicted region mask"
        ),
        "interpretation_limit": (
            "geometric intersection is not proof that pathology is present in that region"
        ),
    }

    llm_evidence = {
        "schema_version": "1.0",
        "case_id": image.image_id,
        "attachments": [
            {"position": 1, "role": "original_chest_xray"},
            {
                "position": 2,
                "role": "visual_overlay_of_detector_attribution_and_segmentation",
            },
        ],
        "model_decision": compact_decision,
        "model_attribution": compact_attribution,
        "anatomical_segmentation": compact_segmentation,
        "computed_spatial_relationships": compact_relationships,
        "knowledge_base": {
            "status": knowledge.status,
            "semantics": (
                "general ontology associations; not patient-specific image findings"
            ),
            "records": compact_knowledge,
        },
    }

    sections = [
        "TASK:\n" + DIAGNOSTIC_QUESTION,
        (
            "INTERPRETATION RULES:\n"
            "- Use only the two images and INPUT_EVIDENCE.\n"
            "- Keep image observations, model outputs, attribution, anatomy, and KB facts separate.\n"
            "- Do not treat model scores as disease probabilities or model evidence as ground truth.\n"
            "- Do not infer a recognized finding, causation, location, or severity from attribution.\n"
            "- Use only the supplied computed relationship for spatial-intersection claims.\n"
            "- State uncertainty and place unsupported/conflicting claims in the designated field.\n"
            "- Do not invent a global classifier score."
        ),
        "INPUT_EVIDENCE:\n" + json.dumps(llm_evidence, indent=2),
        "OUTPUT_SCHEMA:\n" + json.dumps(RESPONSE_TEMPLATE, indent=2),
        "Return only one valid JSON object matching OUTPUT_SCHEMA. Confidence must be 0 to 1.",
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
