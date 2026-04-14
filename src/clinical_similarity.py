from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .data_loader import load_cdss_diagnoses
    from .ml_dataset import load_cdss_base
    from .similarity import DEFAULT_WEIGHTS as DEFAULT_DIAGNOSIS_WEIGHTS
    from .treatment_loader import load_cdss_treatment_features
    from .treatment_similarity import DEFAULT_TREATMENT_WEIGHTS, treatment_jaccard_similarity
except ImportError:  # pragma: no cover
    from data_loader import load_cdss_diagnoses
    from ml_dataset import load_cdss_base
    from similarity import DEFAULT_WEIGHTS as DEFAULT_DIAGNOSIS_WEIGHTS
    from treatment_loader import load_cdss_treatment_features
    from treatment_similarity import DEFAULT_TREATMENT_WEIGHTS, treatment_jaccard_similarity


DEFAULT_CLINICAL_WEIGHTS = {
    "diagnosis": 0.60,
    "treatment": 0.40,
}


def load_clinical_similarity_dataset(db_path: str | Path | None = None) -> pd.DataFrame:
    base_df = load_cdss_base(db_path)
    diagnoses_df = load_cdss_diagnoses(db_path, validate=True, prepare_sets=True)
    treatment_df = load_cdss_treatment_features(db_path, validate=True, prepare_sets=True)

    diagnosis_columns = [
        "hadm_id",
        "primary_diagnosis_icd",
        "primary_icd_3digit",
        "diagnosis_count",
        "unique_icd_count",
        "diagnosis_diversity_ratio",
        "diagnoses_icd_list",
        "diagnoses_3digit_list",
        "icd_version_mix",
        "dx_set",
        "dx_3_set",
    ]
    treatment_columns = [
        "hadm_id",
        "procedure_list",
        "proc_icd_list",
        "proc_icd_versions",
        "proc_count",
        "proc_group_count",
        "surgery_count",
        "rx_drug_list",
        "rx_routes_list",
        "route_diversity",
        "rx_drug_types",
        "rx_total_count",
        "rx_unique_drugs",
        "rx_unique_formulary_drugs",
        "rx_iv_count",
        "rx_oral_count",
        "rx_iv_ratio",
        "treatment_days",
        "avg_rx_duration_days",
        "treatment_complexity_score",
        "treatment_intensity_label",
        "rx_set",
        "proc_set",
    ]

    df = (
        base_df.reset_index(drop=True)
        .merge(diagnoses_df[diagnosis_columns].reset_index(drop=True), on="hadm_id", how="inner")
        .merge(treatment_df[treatment_columns].reset_index(drop=True), on="hadm_id", how="inner")
        .set_index("hadm_id", drop=False)
    )
    return df


def _diagnosis_similarity_score(
    query_primary: Any,
    query_dx_set: set[str],
    query_dx_3_set: set[str],
    candidate_primary: Any,
    candidate_dx_set: set[str],
    candidate_dx_3_set: set[str],
    weights: Mapping[str, float],
) -> float:
    primary_match = float(
        query_primary is not None
        and candidate_primary is not None
        and not pd.isna(query_primary)
        and not pd.isna(candidate_primary)
        and query_primary == candidate_primary
    )
    full_icd_sim = len(query_dx_set & candidate_dx_set) / len(query_dx_set | candidate_dx_set) if (query_dx_set or candidate_dx_set) else 0.0
    three_digit_sim = (
        len(query_dx_3_set & candidate_dx_3_set) / len(query_dx_3_set | candidate_dx_3_set)
        if (query_dx_3_set or candidate_dx_3_set)
        else 0.0
    )
    return (
        weights["primary"] * primary_match
        + weights["full_icd"] * full_icd_sim
        + weights["3digit"] * three_digit_sim
    )


def _treatment_similarity_score(
    query_rx_set: set[str],
    query_proc_set: set[str],
    query_complexity_score: float,
    query_treatment_days: int,
    candidate_rx_set: set[str],
    candidate_proc_set: set[str],
    candidate_complexity_score: float,
    candidate_treatment_days: int,
    weights: Mapping[str, float],
) -> float:
    drug_sim = treatment_jaccard_similarity(query_rx_set, candidate_rx_set)
    proc_sim = treatment_jaccard_similarity(query_proc_set, candidate_proc_set)
    complexity_sim = max(0.0, 1.0 - abs(float(query_complexity_score) - float(candidate_complexity_score)))
    max_duration = max(int(query_treatment_days), int(candidate_treatment_days), 1)
    duration_sim = max(0.0, 1.0 - (abs(int(query_treatment_days) - int(candidate_treatment_days)) / max_duration))
    return (
        weights["drug"] * drug_sim
        + weights["procedure"] * proc_sim
        + weights["complexity"] * complexity_sim
        + weights["duration"] * duration_sim
    )


def find_clinically_similar_patients(
    df: pd.DataFrame,
    query_hadm_id: int,
    k: int = 20,
    same_version_only: bool = True,
    stratify_treatment: bool = True,
    exclude_self: bool = True,
    clinical_weights: Mapping[str, float] | None = None,
    diagnosis_weights: Mapping[str, float] | None = None,
    treatment_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")
    if query_hadm_id not in df.index:
        raise KeyError(f"hadm_id {query_hadm_id} not found in dataframe.")

    active_clinical_weights = clinical_weights or DEFAULT_CLINICAL_WEIGHTS
    active_diagnosis_weights = diagnosis_weights or DEFAULT_DIAGNOSIS_WEIGHTS
    active_treatment_weights = treatment_weights or DEFAULT_TREATMENT_WEIGHTS

    query_row = df.loc[query_hadm_id]
    mask = np.ones(len(df), dtype=bool)
    if exclude_self:
        mask &= df["hadm_id"].to_numpy() != query_hadm_id
    if same_version_only:
        mask &= df["icd_version_mix"].to_numpy(dtype=object) == query_row["icd_version_mix"]
    if stratify_treatment:
        mask &= df["treatment_intensity_label"].to_numpy(dtype=object) == query_row["treatment_intensity_label"]

    candidates = df.loc[mask].reset_index(drop=True)
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "subject_id",
                "diagnosis_similarity",
                "treatment_similarity",
                "combined_similarity",
            ]
        )

    diagnosis_scores = np.empty(len(candidates), dtype=float)
    treatment_scores = np.empty(len(candidates), dtype=float)
    combined_scores = np.empty(len(candidates), dtype=float)

    for idx, (_, row) in enumerate(candidates.iterrows()):
        diagnosis_sim = _diagnosis_similarity_score(
            query_primary=query_row["primary_diagnosis_icd"],
            query_dx_set=query_row["dx_set"],
            query_dx_3_set=query_row["dx_3_set"],
            candidate_primary=row["primary_diagnosis_icd"],
            candidate_dx_set=row["dx_set"],
            candidate_dx_3_set=row["dx_3_set"],
            weights=active_diagnosis_weights,
        )
        treatment_sim = _treatment_similarity_score(
            query_rx_set=query_row["rx_set"],
            query_proc_set=query_row["proc_set"],
            query_complexity_score=query_row["treatment_complexity_score"],
            query_treatment_days=query_row["treatment_days"],
            candidate_rx_set=row["rx_set"],
            candidate_proc_set=row["proc_set"],
            candidate_complexity_score=row["treatment_complexity_score"],
            candidate_treatment_days=row["treatment_days"],
            weights=active_treatment_weights,
        )
        combined = (
            active_clinical_weights["diagnosis"] * diagnosis_sim
            + active_clinical_weights["treatment"] * treatment_sim
        )
        diagnosis_scores[idx] = diagnosis_sim
        treatment_scores[idx] = treatment_sim
        combined_scores[idx] = combined

    results = candidates[
        [
            "hadm_id",
            "subject_id",
            "primary_diagnosis_icd",
            "treatment_intensity_label",
            "mortality",
            "los_days",
        ]
    ].copy()
    results["diagnosis_similarity"] = diagnosis_scores
    results["treatment_similarity"] = treatment_scores
    results["combined_similarity"] = combined_scores
    return results.sort_values("combined_similarity", ascending=False).head(k).reset_index(drop=True)


def recommend_treatments(
    df: pd.DataFrame,
    query_hadm_id: int,
    k: int = 20,
    top_n: int = 10,
    use_clinical_similarity: bool = True,
) -> pd.DataFrame:
    if use_clinical_similarity:
        similar_patients = find_clinically_similar_patients(df=df, query_hadm_id=query_hadm_id, k=k)
        similarity_column = "combined_similarity"
    else:
        from .treatment_similarity import find_similar_treatment_patients  # local import to avoid cycles

        similar_patients = find_similar_treatment_patients(df=df, query_hadm_id=query_hadm_id, k=k)
        similarity_column = "composite_similarity"

    if similar_patients.empty:
        return pd.DataFrame(columns=["drug", "frequency", "weighted_frequency", "mean_similarity", "mortality_rate", "avg_los_days"])

    records: list[dict[str, Any]] = []
    for _, row in similar_patients.iterrows():
        hadm_id = int(row["hadm_id"])
        drug_list = df.loc[hadm_id, "rx_drug_list"]
        for drug in drug_list:
            records.append(
                {
                    "drug": drug,
                    "hadm_id": hadm_id,
                    "similarity": float(row[similarity_column]),
                    "mortality": float(df.loc[hadm_id, "mortality"]) if "mortality" in df.columns else np.nan,
                    "los_days": float(df.loc[hadm_id, "los_days"]) if "los_days" in df.columns else np.nan,
                }
            )

    rec_df = pd.DataFrame(records)
    recommendations = (
        rec_df.groupby("drug")
        .agg(
            frequency=("hadm_id", "nunique"),
            weighted_frequency=("similarity", "sum"),
            mean_similarity=("similarity", "mean"),
            mortality_rate=("mortality", "mean"),
            avg_los_days=("los_days", "mean"),
        )
        .sort_values(["weighted_frequency", "frequency", "mean_similarity"], ascending=False)
        .head(top_n)
        .reset_index()
    )
    return recommendations
