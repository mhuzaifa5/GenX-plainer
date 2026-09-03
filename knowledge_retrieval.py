"""Retrieve grounded pneumonia knowledge from the local curated CSV."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from classifier import ClassificationResult
from segmentation import SegmentationResult
from xai import AttributionResult


SYNONYMS = {
    "upper": {"apical"},
    "lower": {"basilar", "base", "retrocardiac"},
    "middle": {"perihilar"},
    "pneumonia": {"airspace", "opacity", "consolidation", "infiltration"},
}
ANATOMY_TERMS = {
    "upper", "middle", "lower", "apical", "basilar", "base",
    "retrocardiac", "perihilar", "right", "left", "medial", "lateral",
}
THORACIC_TERMS = {
    "lung", "lungs", "pulmonary", "pleural", "opacity", "opacities",
    "airspace", "cavitary", "cavity", "pneumatocele", "pneumothorax",
    "hemithorax", "perihilar", "retrocardiac", "bronchiectasis", "ards",
    "diaphragm", "cardiophrenic", "mediastinal", "mediastinitis",
}


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z]+", str(value).lower()))


@dataclass(frozen=True)
class RetrievedKnowledge:
    disease_label: str
    relation: str
    finding_label: str
    source: str
    disease_rgo_uri: str
    finding_rgo_uri: str
    relation_rgo_uri: str
    radlex_uri: str
    radlex_label: str
    mapping_status: str
    retrieval_score: float
    matched_terms: list[str]
    retrieval_reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "disease_label": self.disease_label,
            "relation": self.relation,
            "finding_label": self.finding_label,
            "source": self.source,
            "disease_rgo_uri": self.disease_rgo_uri,
            "finding_rgo_uri": self.finding_rgo_uri,
            "relation_rgo_uri": self.relation_rgo_uri,
            "radlex_uri": self.radlex_uri,
            "radlex_label": self.radlex_label,
            "mapping_status": self.mapping_status,
            "retrieval_score": self.retrieval_score,
            "matched_terms": self.matched_terms,
            "retrieval_reasons": self.retrieval_reasons,
        }


@dataclass(frozen=True)
class KnowledgeRetrievalResult:
    status: str
    query_terms: list[str]
    selected_attribution_regions: list[str]
    box_overlap_regions: list[str]
    records: list[RetrievedKnowledge]
    kb_warnings: list[str]
    grounding_note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "query_terms": self.query_terms,
            "selected_attribution_regions": self.selected_attribution_regions,
            "box_overlap_regions": self.box_overlap_regions,
            "records": [record.as_dict() for record in self.records],
            "kb_warnings": self.kb_warnings,
            "grounding_note": self.grounding_note,
        }


class PneumoniaKnowledgeRetriever:
    """Small-KB lexical retriever with auditable, evidence-derived queries."""

    def __init__(self, kb_path: str | Path, top_k: int = 5) -> None:
        self.kb_path = Path(kb_path).expanduser().resolve()
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.top_k = top_k
        with self.kb_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Knowledge base is empty: {self.kb_path}")
        required = {"disease_label", "relation", "finding_label", "source"}
        missing = required - set(self.rows[0])
        if missing:
            raise ValueError(f"Knowledge base is missing columns: {sorted(missing)}")
        self.kb_warnings = self._validate_rows()

    def _validate_rows(self) -> list[str]:
        warnings: list[str] = []
        for number, row in enumerate(self.rows, start=2):
            relation = row.get("relation", "").strip()
            relation_uri = row.get("relation_rgo_uri", "").strip().rstrip("/")
            if relation and relation_uri and relation_uri.rsplit("/", 1)[-1] != relation:
                warnings.append(
                    f"Row {number}: relation {relation!r} conflicts with URI {relation_uri!r}."
                )
            if row.get("mapping_status", "").strip().lower() == "ambiguous":
                warnings.append(
                    f"Row {number}: {row.get('finding_label', '')!r} has an ambiguous mapping."
                )
        return warnings

    @staticmethod
    def _top_attribution_regions(
        attribution: AttributionResult,
        segmentation: SegmentationResult,
        count: int = 3,
    ) -> list[str]:
        ranked: list[tuple[str, float]] = []
        for name, mask in segmentation.regions.items():
            if name == "heart":
                continue
            selected = np.asarray(mask) > 0
            if selected.any():
                ranked.append((name, float(attribution.absolute_map[selected].mean())))
        return [name for name, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:count]]

    @staticmethod
    def _box_overlap_regions(
        prediction: ClassificationResult,
        segmentation: SegmentationResult,
    ) -> list[str]:
        height, width = segmentation.class_map.shape
        overlaps: set[str] = set()
        for detection in prediction.detections:
            if detection.label_id != 1 or detection.score < prediction.detection_threshold:
                continue
            x1, y1, x2, y2 = detection.box
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            for name, mask in segmentation.regions.items():
                if name != "heart" and np.asarray(mask)[y1:y2, x1:x2].any():
                    overlaps.add(name)
        return sorted(overlaps)

    def retrieve(
        self,
        prediction: ClassificationResult,
        attribution: AttributionResult,
        segmentation: SegmentationResult,
    ) -> KnowledgeRetrievalResult:
        note = (
            "Retrieved records are general KB associations, not findings established "
            "as present in this patient's image."
        )
        if prediction.predicted_class == "normal":
            return KnowledgeRetrievalResult(
                status="not_retrieved_for_normal_prediction",
                query_terms=[],
                selected_attribution_regions=[],
                box_overlap_regions=[],
                records=[],
                kb_warnings=self.kb_warnings,
                grounding_note=note,
            )
        if prediction.predicted_class != "pneumonia":
            raise ValueError(f"Unsupported predicted class: {prediction.predicted_class!r}")

        attribution_regions = self._top_attribution_regions(attribution, segmentation)
        overlap_regions = self._box_overlap_regions(prediction, segmentation)
        query_terms = _tokens("pneumonia " + " ".join(attribution_regions + overlap_regions))
        for term in list(query_terms):
            query_terms.update(SYNONYMS.get(term, set()))

        ranked: list[RetrievedKnowledge] = []
        for row in self.rows:
            if row.get("disease_label", "").strip().lower() != "pneumonia":
                continue
            finding_tokens = _tokens(row.get("finding_label", ""))
            if not finding_tokens.intersection(THORACIC_TERMS):
                continue
            matched = sorted(finding_tokens.intersection(query_terms))
            anatomical = sorted(set(matched).intersection(ANATOMY_TERMS))
            score = 3.0 + 2.0 * len(matched) + 2.0 * len(anatomical)
            reasons = ["exact disease match"]
            if matched:
                reasons.append("query overlap: " + ", ".join(matched))
            if anatomical:
                reasons.append("anatomical match: " + ", ".join(anatomical))
            mapping = row.get("mapping_status", "").strip().lower()
            if mapping == "mapped":
                score += 1.0
                reasons.append("mapped ontology term")
            elif mapping == "ambiguous":
                score -= 1.0
                reasons.append("ambiguous ontology mapping penalty")
            if not matched:
                continue
            ranked.append(
                RetrievedKnowledge(
                    disease_label=row.get("disease_label", ""),
                    relation=row.get("relation", ""),
                    finding_label=row.get("finding_label", ""),
                    source=row.get("source", ""),
                    disease_rgo_uri=row.get("disease_rgo_uri", ""),
                    finding_rgo_uri=row.get("finding_rgo_uri", ""),
                    relation_rgo_uri=row.get("relation_rgo_uri", ""),
                    radlex_uri=row.get("radlex_uri", ""),
                    radlex_label=row.get("radlex_label", ""),
                    mapping_status=row.get("mapping_status", ""),
                    retrieval_score=score,
                    matched_terms=matched,
                    retrieval_reasons=reasons,
                )
            )

        ranked.sort(key=lambda item: (-item.retrieval_score, item.finding_label))
        return KnowledgeRetrievalResult(
            status="retrieved",
            query_terms=sorted(query_terms),
            selected_attribution_regions=attribution_regions,
            box_overlap_regions=overlap_regions,
            records=ranked[: self.top_k],
            kb_warnings=self.kb_warnings,
            grounding_note=note,
        )
