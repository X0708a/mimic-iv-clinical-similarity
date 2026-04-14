from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

try:
    from .data_loader import prepare_similarity_columns
except ImportError:  # pragma: no cover
    from data_loader import prepare_similarity_columns


DEFAULT_WEIGHTS = {
    "primary": 0.40,
    "full_icd": 0.35,
    "3digit": 0.25,
}


def _is_missing_scalar(value: Any) -> bool:
    return value is None or pd.isna(value)


def _to_code_set(values: Iterable[str] | set[str] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, set):
        return values
    return set(values)


def _jaccard_set_similarity(set_a: set[str], set_b: set[str]) -> float:
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size


def _extract_set(patient: Mapping[str, Any], set_key: str, list_key: str) -> set[str]:
    values = patient.get(set_key)
    if values is not None:
        return _to_code_set(values)
    return _to_code_set(patient.get(list_key))


def _primary_match_score(primary_a: Any, primary_b: Any) -> float:
    if _is_missing_scalar(primary_a) or _is_missing_scalar(primary_b):
        return 0.0
    return float(primary_a == primary_b)


def _weighted_similarity_from_parts(
    query_primary: Any,
    query_dx_set: set[str],
    query_dx_3_set: set[str],
    candidate_primary: Any,
    candidate_dx_set: set[str],
    candidate_dx_3_set: set[str],
    weights: Mapping[str, float],
) -> float:
    primary_sim = _primary_match_score(query_primary, candidate_primary)
    full_icd_sim = _jaccard_set_similarity(query_dx_set, candidate_dx_set)
    three_digit_sim = _jaccard_set_similarity(query_dx_3_set, candidate_dx_3_set)
    return (
        weights["primary"] * primary_sim
        + weights["full_icd"] * full_icd_sim
        + weights["3digit"] * three_digit_sim
    )


def jaccard_similarity(a: Iterable[str] | set[str] | None, b: Iterable[str] | set[str] | None) -> float:
    return _jaccard_set_similarity(_to_code_set(a), _to_code_set(b))


def baseline_patient_similarity(
    patient_a: Mapping[str, Any] | pd.Series,
    patient_b: Mapping[str, Any] | pd.Series,
    weights: Mapping[str, float] | None = None,
) -> float:
    active_weights = weights or DEFAULT_WEIGHTS

    return _weighted_similarity_from_parts(
        query_primary=patient_a.get("primary_diagnosis_icd"),
        query_dx_set=_extract_set(patient_a, "dx_set", "diagnoses_icd_list"),
        query_dx_3_set=_extract_set(patient_a, "dx_3_set", "diagnoses_3digit_list"),
        candidate_primary=patient_b.get("primary_diagnosis_icd"),
        candidate_dx_set=_extract_set(patient_b, "dx_set", "diagnoses_icd_list"),
        candidate_dx_3_set=_extract_set(patient_b, "dx_3_set", "diagnoses_3digit_list"),
        weights=active_weights,
    )


def _resolve_query_position(df: pd.DataFrame, query_hadm_id: int) -> int:
    if df.index.name == "hadm_id" and df.index.is_unique:
        if query_hadm_id not in df.index:
            raise KeyError(f"hadm_id {query_hadm_id} not found in dataframe.")
        return int(df.index.get_loc(query_hadm_id))

    matches = np.flatnonzero(df["hadm_id"].to_numpy() == query_hadm_id)
    if len(matches) == 0:
        raise KeyError(f"hadm_id {query_hadm_id} not found in dataframe.")
    return int(matches[0])


def find_similar_patients(
    df: pd.DataFrame,
    query_hadm_id: int,
    k: int = 10,
    same_version_only: bool = False,
    exclude_self: bool = False,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")

    scored = score_all_patients(
        df=df,
        query_hadm_id=query_hadm_id,
        same_version_only=same_version_only,
        exclude_self=exclude_self,
        weights=weights,
    )
    if scored.empty:
        return scored

    top_k = min(int(k), len(scored))
    scores = scored["similarity_score"].to_numpy()
    if top_k == len(scored):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return scored.iloc[top_indices].reset_index(drop=True)


def score_all_patients(
    df: pd.DataFrame,
    query_hadm_id: int,
    same_version_only: bool = False,
    exclude_self: bool = False,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hadm_id", "subject_id", "primary_diagnosis_icd", "similarity_score"])

    prepare_similarity_columns(df)
    active_weights = weights or DEFAULT_WEIGHTS
    query_position = _resolve_query_position(df, query_hadm_id)

    hadm_ids = df["hadm_id"].to_numpy()
    subject_ids = df["subject_id"].to_numpy()
    primary_dx = df["primary_diagnosis_icd"].to_numpy(dtype=object)
    icd_version_mix = df["icd_version_mix"].to_numpy(dtype=object)
    diagnosis_counts = df["diagnosis_count"].to_numpy()
    unique_icd_counts = df["unique_icd_count"].to_numpy()
    dx_sets = df["dx_set"].to_numpy(dtype=object)
    dx_3_sets = df["dx_3_set"].to_numpy(dtype=object)

    query_primary = primary_dx[query_position]
    query_icd_version_mix = icd_version_mix[query_position]
    query_dx_set = dx_sets[query_position]
    query_dx_3_set = dx_3_sets[query_position]

    mask = np.ones(len(df), dtype=bool)
    if exclude_self:
        mask[query_position] = False
    if same_version_only:
        mask &= icd_version_mix == query_icd_version_mix

    candidate_count = int(mask.sum())
    if candidate_count == 0:
        return pd.DataFrame(columns=["hadm_id", "subject_id", "primary_diagnosis_icd", "similarity_score"])

    scores = np.fromiter(
        (
            _weighted_similarity_from_parts(
                query_primary=query_primary,
                query_dx_set=query_dx_set,
                query_dx_3_set=query_dx_3_set,
                candidate_primary=candidate_primary,
                candidate_dx_set=candidate_dx_set,
                candidate_dx_3_set=candidate_dx_3_set,
                weights=active_weights,
            )
            for candidate_primary, candidate_dx_set, candidate_dx_3_set in zip(
                primary_dx[mask],
                dx_sets[mask],
                dx_3_sets[mask],
            )
        ),
        dtype=float,
        count=candidate_count,
    )

    return pd.DataFrame(
        {
            "hadm_id": hadm_ids[mask],
            "subject_id": subject_ids[mask],
            "primary_diagnosis_icd": primary_dx[mask],
            "diagnosis_count": diagnosis_counts[mask],
            "unique_icd_count": unique_icd_counts[mask],
            "icd_version_mix": icd_version_mix[mask],
            "similarity_score": scores,
        }
    )
