from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

try:
    from .treatment_loader import prepare_treatment_similarity_columns
except ImportError:  # pragma: no cover
    from treatment_loader import prepare_treatment_similarity_columns


DEFAULT_TREATMENT_WEIGHTS = {
    "drug": 0.40,
    "procedure": 0.35,
    "complexity": 0.15,
    "duration": 0.10,
}


def _coerce_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _coerce_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _resolve_query_position(df: pd.DataFrame, query_hadm_id: int) -> int:
    if df.index.name == "hadm_id" and df.index.is_unique:
        if query_hadm_id not in df.index:
            raise KeyError(f"hadm_id {query_hadm_id} not found in dataframe.")
        return int(df.index.get_loc(query_hadm_id))

    matches = np.flatnonzero(df["hadm_id"].to_numpy() == query_hadm_id)
    if len(matches) == 0:
        raise KeyError(f"hadm_id {query_hadm_id} not found in dataframe.")
    return int(matches[0])


def _empty_aware_jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _duration_similarity(days_a: Any, days_b: Any) -> float:
    duration_a = max(int(days_a or 0), 0)
    duration_b = max(int(days_b or 0), 0)
    max_duration = max(duration_a, duration_b, 1)
    return max(0.0, 1.0 - (abs(duration_a - duration_b) / max_duration))


def _complexity_similarity(score_a: Any, score_b: Any) -> float:
    return max(0.0, 1.0 - abs(float(score_a) - float(score_b)))


def treatment_jaccard_similarity(a: set[str] | list[str] | None, b: set[str] | list[str] | None) -> float:
    set_a = a if isinstance(a, set) else set(a or [])
    set_b = b if isinstance(b, set) else set(b or [])
    return _empty_aware_jaccard(set_a, set_b)


def _treatment_similarity_components(
    rx_set_a: set[str],
    proc_set_a: set[str],
    complexity_score_a: float,
    treatment_days_a: int,
    rx_set_b: set[str],
    proc_set_b: set[str],
    complexity_score_b: float,
    treatment_days_b: int,
    weights: Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    drug_sim = _empty_aware_jaccard(rx_set_a, rx_set_b)
    proc_sim = _empty_aware_jaccard(proc_set_a, proc_set_b)
    complexity_sim = _complexity_similarity(complexity_score_a, complexity_score_b)
    duration_sim = _duration_similarity(treatment_days_a, treatment_days_b)
    composite = (
        weights["drug"] * drug_sim
        + weights["procedure"] * proc_sim
        + weights["complexity"] * complexity_sim
        + weights["duration"] * duration_sim
    )
    return drug_sim, proc_sim, complexity_sim, duration_sim, composite


def baseline_treatment_similarity(
    patient_a: Mapping[str, Any] | pd.Series,
    patient_b: Mapping[str, Any] | pd.Series,
    weights: Mapping[str, float] | None = None,
) -> float:
    active_weights = weights or DEFAULT_TREATMENT_WEIGHTS
    _, _, _, _, composite = _treatment_similarity_components(
        rx_set_a=patient_a.get("rx_set", set(patient_a.get("rx_drug_list", []))),
        proc_set_a=patient_a.get("proc_set", set(patient_a.get("proc_icd_list", []))),
        complexity_score_a=_coerce_float(patient_a.get("treatment_complexity_score", 0.0)),
        treatment_days_a=_coerce_int(patient_a.get("treatment_days", 0)),
        rx_set_b=patient_b.get("rx_set", set(patient_b.get("rx_drug_list", []))),
        proc_set_b=patient_b.get("proc_set", set(patient_b.get("proc_icd_list", []))),
        complexity_score_b=_coerce_float(patient_b.get("treatment_complexity_score", 0.0)),
        treatment_days_b=_coerce_int(patient_b.get("treatment_days", 0)),
        weights=active_weights,
    )
    return composite


def score_all_treatment_patients(
    df: pd.DataFrame,
    query_hadm_id: int,
    stratify: bool = True,
    exclude_self: bool = True,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["hadm_id", "subject_id", "treatment_intensity_label", "drug_sim", "proc_sim", "composite_similarity"]
        )

    prepare_treatment_similarity_columns(df)
    active_weights = weights or DEFAULT_TREATMENT_WEIGHTS
    query_position = _resolve_query_position(df, query_hadm_id)

    hadm_ids = df["hadm_id"].to_numpy()
    subject_ids = df["subject_id"].to_numpy()
    intensity_labels = df["treatment_intensity_label"].to_numpy(dtype=object)
    rx_sets = df["rx_set"].to_numpy(dtype=object)
    proc_sets = df["proc_set"].to_numpy(dtype=object)
    complexity_scores = df["treatment_complexity_score"].to_numpy(dtype=float)
    treatment_days = df["treatment_days"].to_numpy(dtype=int)
    rx_unique_drugs = df["rx_unique_drugs"].to_numpy(dtype=int)
    proc_count = df["proc_count"].to_numpy(dtype=int)

    query_label = intensity_labels[query_position]
    mask = np.ones(len(df), dtype=bool)
    if exclude_self:
        mask[query_position] = False
    if stratify:
        mask &= intensity_labels == query_label

    candidate_count = int(mask.sum())
    if candidate_count == 0:
        return pd.DataFrame(
            columns=["hadm_id", "subject_id", "treatment_intensity_label", "drug_sim", "proc_sim", "composite_similarity"]
        )

    drug_scores = np.empty(candidate_count, dtype=float)
    proc_scores = np.empty(candidate_count, dtype=float)
    complexity_matches = np.empty(candidate_count, dtype=float)
    duration_matches = np.empty(candidate_count, dtype=float)
    composite_scores = np.empty(candidate_count, dtype=float)

    query_rx_set = rx_sets[query_position]
    query_proc_set = proc_sets[query_position]
    query_complexity = complexity_scores[query_position]
    query_days = treatment_days[query_position]

    for idx, (candidate_rx_set, candidate_proc_set, candidate_complexity, candidate_days) in enumerate(
        zip(
            rx_sets[mask],
            proc_sets[mask],
            complexity_scores[mask],
            treatment_days[mask],
        )
    ):
        drug_sim, proc_sim, complexity_sim, duration_sim, composite = _treatment_similarity_components(
            rx_set_a=query_rx_set,
            proc_set_a=query_proc_set,
            complexity_score_a=query_complexity,
            treatment_days_a=query_days,
            rx_set_b=candidate_rx_set,
            proc_set_b=candidate_proc_set,
            complexity_score_b=candidate_complexity,
            treatment_days_b=candidate_days,
            weights=active_weights,
        )
        drug_scores[idx] = drug_sim
        proc_scores[idx] = proc_sim
        complexity_matches[idx] = complexity_sim
        duration_matches[idx] = duration_sim
        composite_scores[idx] = composite

    return pd.DataFrame(
        {
            "hadm_id": hadm_ids[mask],
            "subject_id": subject_ids[mask],
            "treatment_intensity_label": intensity_labels[mask],
            "rx_unique_drugs": rx_unique_drugs[mask],
            "proc_count": proc_count[mask],
            "drug_sim": drug_scores,
            "proc_sim": proc_scores,
            "complexity_sim": complexity_matches,
            "duration_sim": duration_matches,
            "composite_similarity": composite_scores,
        }
    )


def find_similar_treatment_patients(
    df: pd.DataFrame,
    query_hadm_id: int,
    k: int = 20,
    stratify: bool = True,
    exclude_self: bool = True,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")

    scored = score_all_treatment_patients(
        df=df,
        query_hadm_id=query_hadm_id,
        stratify=stratify,
        exclude_self=exclude_self,
        weights=weights,
    )
    if scored.empty:
        return scored

    top_k = min(int(k), len(scored))
    scores = scored["composite_similarity"].to_numpy()
    if top_k == len(scored):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return scored.iloc[top_indices].reset_index(drop=True)
