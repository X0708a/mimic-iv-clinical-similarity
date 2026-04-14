from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


EPSILON_LOS = 1e-6
DEFAULT_NEIGHBORS = 7


@dataclass(frozen=True)
class CDSSConfig:
    db_path: Path
    n_neighbors: int = DEFAULT_NEIGHBORS


class SimilarityBasedCDSS:
    def __init__(self, db_path: str | Path | None = None, n_neighbors: int = DEFAULT_NEIGHBORS):
        project_dir = Path(__file__).resolve().parent
        resolved_db_path = Path(db_path) if db_path else project_dir / "mimic.db"
        self.config = CDSSConfig(db_path=resolved_db_path, n_neighbors=max(1, int(n_neighbors)))

        self.dataframe: pd.DataFrame | None = None
        self.feature_matrix: sparse.csr_matrix | None = None
        self.model: NearestNeighbors | None = None
        self.treatment_df: pd.DataFrame | None = None
        self.treatments_by_hadm: dict[int, list[dict[str, Any]]] = {}
        self.numeric_features = [
            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "diagnosis_count",
            "unique_icd_count",
            "diagnosis_diversity_ratio",
        ]
        self.excluded_similarity_features = ["los_days", "mortality", "discharge_location"]

        self._gender_encoder: OneHotEncoder | None = None
        self._race_encoder: OneHotEncoder | None = None
        self._icd_encoder: MultiLabelBinarizer | None = None
        self._icd_group_encoder: MultiLabelBinarizer | None = None
        self._numeric_scaler: StandardScaler | None = None
        self._numeric_medians: dict[str, float] = {}

    def fit(self) -> "SimilarityBasedCDSS":
        merged = self._load_merged_dataframe()
        self.dataframe = merged
        self.treatment_df = self._load_treatment_dataframe()
        self.treatments_by_hadm = self._build_treatment_lookup(self.treatment_df)
        self.feature_matrix = self._build_feature_matrix(merged)
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(self.config.n_neighbors, max(1, len(merged) - 1)),
            n_jobs=-1,
        )
        self.model.fit(self.feature_matrix)
        return self

    def get_similar_patients(self, patient_index: int, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        assert self.dataframe is not None
        assert self.feature_matrix is not None
        assert self.model is not None

        self._validate_patient_index(patient_index)
        requested_neighbors = top_k if top_k is not None else self.config.n_neighbors
        requested_neighbors = max(1, int(requested_neighbors))
        max_neighbors = min(len(self.dataframe), requested_neighbors + 1)

        distances, indices = self.model.kneighbors(
            self.feature_matrix[patient_index],
            n_neighbors=max_neighbors,
            return_distance=True,
        )

        neighbor_indices = []
        neighbor_distances = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == patient_index:
                continue
            neighbor_indices.append(int(idx))
            neighbor_distances.append(float(distance))
            if len(neighbor_indices) >= requested_neighbors:
                break

        return np.asarray(neighbor_indices, dtype=int), np.asarray(neighbor_distances, dtype=float)

    def recommend_for_patient(self, patient_index: int, top_k: int | None = None) -> dict[str, Any]:
        self._ensure_fitted()
        assert self.dataframe is not None

        neighbor_indices, distances = self.get_similar_patients(patient_index, top_k=top_k)
        similarities = np.clip(1.0 - distances, 0.0, 1.0)

        query_row = self.dataframe.iloc[patient_index]
        similar_patients = []
        treatment_records = []

        for neighbor_idx, distance, similarity in zip(neighbor_indices, distances, similarities):
            neighbor_row = self.dataframe.iloc[int(neighbor_idx)]
            shared_diagnoses = self._shared_items(query_row["diagnoses_icd_list"], neighbor_row["diagnoses_icd_list"])
            shared_groups = self._shared_items(
                query_row["diagnoses_3digit_list"], neighbor_row["diagnoses_3digit_list"]
            )
            feature_deltas = self._feature_similarity_summary(query_row, neighbor_row)

            similar_patients.append(
                {
                    "patient_index": int(neighbor_idx),
                    "subject_id": int(neighbor_row["subject_id"]),
                    "hadm_id": int(neighbor_row["hadm_id"]),
                    "distance": float(distance),
                    "similarity": float(similarity),
                    "los_days": float(neighbor_row["los_days"]),
                    "shared_diagnoses": shared_diagnoses[:10],
                    "shared_icd_groups": shared_groups[:10],
                    "key_similar_features": feature_deltas,
                }
            )

            treatment_records.extend(
                self._treatment_records_for_neighbor(neighbor_row, float(similarity), int(neighbor_idx))
            )

        recommended_treatments = self._score_treatments(treatment_records)

        return {
            "similar_patients": similar_patients,
            "recommended_treatments": recommended_treatments,
            "explanation": {
                "query_patient": {
                    "patient_index": int(patient_index),
                    "subject_id": int(query_row["subject_id"]),
                    "hadm_id": int(query_row["hadm_id"]),
                    "age": float(query_row["age"]),
                    "gender": query_row["gender"],
                    "race": query_row["race"],
                    "bmi": self._safe_float(query_row["bmi"]),
                    "systolic_bp": self._safe_float(query_row["systolic_bp"]),
                    "diastolic_bp": self._safe_float(query_row["diastolic_bp"]),
                    "diagnoses": self._normalize_list(query_row["diagnoses_icd_list"]),
                    "diagnosis_groups": self._normalize_list(query_row["diagnoses_3digit_list"]),
                },
                "neighbor_count": len(similar_patients),
                "similarity_metric": "cosine distance via sklearn NearestNeighbors",
                "los_used_for_scoring_only": True,
            },
        }

    def get_similarities_for_profile(
        self,
        patient_profile: dict[str, Any],
        top_k: int | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_fitted()
        assert self.dataframe is not None
        assert self.feature_matrix is not None
        assert self.model is not None

        requested_neighbors = top_k if top_k is not None else self.config.n_neighbors
        requested_neighbors = max(1, int(requested_neighbors))
        query_profile = self._prepare_profile_row(patient_profile)
        query_frame = pd.DataFrame([query_profile])
        query_matrix = self._transform_rows(query_frame)

        distances, indices = self.model.kneighbors(
            query_matrix,
            n_neighbors=min(requested_neighbors, len(self.dataframe)),
            return_distance=True,
        )
        similarities = np.clip(1.0 - distances[0], 0.0, 1.0)
        return query_profile, indices[0].astype(int), distances[0].astype(float), similarities.astype(float)

    def recommend_for_profile(
        self,
        patient_profile: dict[str, Any],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_fitted()
        assert self.dataframe is not None

        query_profile, neighbor_indices, distances, similarities = self.get_similarities_for_profile(
            patient_profile=patient_profile,
            top_k=top_k,
        )

        query_series = pd.Series(query_profile)
        similar_patients = []
        treatment_records = []

        for neighbor_idx, distance, similarity in zip(neighbor_indices, distances, similarities):
            neighbor_row = self.dataframe.iloc[int(neighbor_idx)]
            shared_diagnoses = self._shared_items(query_series["diagnoses_icd_list"], neighbor_row["diagnoses_icd_list"])
            shared_groups = self._shared_items(
                query_series["diagnoses_3digit_list"], neighbor_row["diagnoses_3digit_list"]
            )
            feature_deltas = self._feature_similarity_summary(query_series, neighbor_row)

            similar_patients.append(
                {
                    "patient_index": int(neighbor_idx),
                    "subject_id": int(neighbor_row["subject_id"]),
                    "hadm_id": int(neighbor_row["hadm_id"]),
                    "distance": float(distance),
                    "similarity": float(similarity),
                    "los_days": float(neighbor_row["los_days"]),
                    "shared_diagnoses": shared_diagnoses[:10],
                    "shared_icd_groups": shared_groups[:10],
                    "key_similar_features": feature_deltas,
                }
            )
            treatment_records.extend(
                self._treatment_records_for_neighbor(neighbor_row, float(similarity), int(neighbor_idx))
            )

        recommended_treatments = self._score_treatments(treatment_records)

        return {
            "similar_patients": similar_patients,
            "recommended_treatments": recommended_treatments,
            "explanation": {
                "query_patient": {
                    "age": float(query_profile["age"]),
                    "gender": query_profile["gender"],
                    "race": query_profile["race"],
                    "bmi": self._safe_float(query_profile["bmi"]),
                    "systolic_bp": self._safe_float(query_profile["systolic_bp"]),
                    "diastolic_bp": self._safe_float(query_profile["diastolic_bp"]),
                    "diagnoses": self._normalize_list(query_profile["diagnoses_icd_list"]),
                    "diagnosis_groups": self._normalize_list(query_profile["diagnoses_3digit_list"]),
                },
                "neighbor_count": len(similar_patients),
                "similarity_metric": "cosine distance via sklearn NearestNeighbors",
                "los_used_for_scoring_only": True,
            },
        }

    def _load_merged_dataframe(self) -> pd.DataFrame:
        with duckdb.connect(str(self.config.db_path), read_only=True) as con:
            merged = con.execute(
                """
                SELECT
                    b.subject_id,
                    b.hadm_id,
                    b.age,
                    b.gender,
                    b.race,
                    b.bmi,
                    b.systolic_bp,
                    b.diastolic_bp,
                    b.los_days,
                    b.mortality,
                    b.discharge_location,
                    COALESCE(d.primary_diagnosis_icd, '') AS primary_diagnosis_icd,
                    COALESCE(d.primary_diagnosis_title, '') AS primary_diagnosis_title,
                    COALESCE(d.primary_icd_3digit, '') AS primary_icd_3digit,
                    COALESCE(d.diagnosis_count, 0) AS diagnosis_count,
                    COALESCE(d.unique_icd_count, 0) AS unique_icd_count,
                    COALESCE(d.icd_version_mix, 'unknown') AS icd_version_mix,
                    COALESCE(d.diagnosis_diversity_ratio, 0.0) AS diagnosis_diversity_ratio,
                    COALESCE(d.diagnoses_icd_list, []) AS diagnoses_icd_list,
                    COALESCE(d.diagnoses_3digit_list, []) AS diagnoses_3digit_list
                FROM cdss_base b
                LEFT JOIN cdss_diagnoses d
                    ON b.subject_id = d.subject_id
                   AND b.hadm_id = d.hadm_id
                ORDER BY b.hadm_id
                """
            ).df()

        for col in ["diagnoses_icd_list", "diagnoses_3digit_list"]:
            merged[col] = merged[col].apply(self._normalize_list)

        merged["gender"] = merged["gender"].fillna("unknown").astype(str)
        merged["race"] = merged["race"].fillna("unknown").astype(str)
        merged["primary_diagnosis_icd"] = merged["primary_diagnosis_icd"].fillna("").astype(str)
        merged["primary_diagnosis_title"] = merged["primary_diagnosis_title"].fillna("").astype(str)
        merged["primary_icd_3digit"] = merged["primary_icd_3digit"].fillna("").astype(str)

        for column in self.numeric_features + ["los_days"]:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

        merged["mortality"] = pd.to_numeric(merged["mortality"], errors="coerce").fillna(0).astype(int)
        merged["discharge_location"] = merged["discharge_location"].fillna("unknown").astype(str)
        return merged

    def _load_treatment_dataframe(self) -> pd.DataFrame:
        with duckdb.connect(str(self.config.db_path), read_only=True) as con:
            if self._table_exists(con, "cdss_treatment"):
                treatment_df = con.execute(
                    """
                    SELECT
                        subject_id,
                        hadm_id,
                        treatment_source,
                        treatment_key,
                        treatment_name,
                        treatment_count
                    FROM cdss_treatment
                    WHERE hadm_id IS NOT NULL
                    """
                ).df()
            else:
                treatment_df = con.execute(
                    """
                    WITH prescription_treatments AS (
                        SELECT
                            subject_id,
                            hadm_id,
                            'prescription' AS treatment_source,
                            UPPER(TRIM(drug)) AS treatment_key,
                            ANY_VALUE(drug) AS treatment_name,
                            COUNT(*) AS treatment_count
                        FROM prescriptions
                        WHERE hadm_id IS NOT NULL
                          AND drug IS NOT NULL
                          AND TRIM(drug) <> ''
                        GROUP BY 1, 2, 3, 4
                    ),
                    procedure_treatments AS (
                        SELECT
                            p.subject_id,
                            p.hadm_id,
                            'procedure' AS treatment_source,
                            p.icd_version || ':' || TRIM(p.icd_code) AS treatment_key,
                            COALESCE(dx.long_title, TRIM(p.icd_code)) AS treatment_name,
                            COUNT(*) AS treatment_count
                        FROM procedures_icd p
                        LEFT JOIN d_icd_procedures dx
                            ON TRIM(p.icd_code) = TRIM(dx.icd_code)
                           AND p.icd_version = dx.icd_version
                        WHERE p.hadm_id IS NOT NULL
                          AND p.icd_code IS NOT NULL
                          AND TRIM(p.icd_code) <> ''
                        GROUP BY 1, 2, 3, 4, 5
                    )
                    SELECT * FROM prescription_treatments
                    UNION ALL
                    SELECT * FROM procedure_treatments
                    """
                ).df()

        if treatment_df.empty:
            return pd.DataFrame(
                columns=[
                    "subject_id",
                    "hadm_id",
                    "treatment_source",
                    "treatment_key",
                    "treatment_name",
                    "treatment_count",
                ]
            )

        treatment_df["treatment_name"] = treatment_df["treatment_name"].fillna("unknown").astype(str)
        treatment_df["treatment_key"] = treatment_df["treatment_key"].fillna("unknown").astype(str)
        treatment_df["treatment_source"] = treatment_df["treatment_source"].fillna("unknown").astype(str)
        treatment_df["treatment_count"] = pd.to_numeric(treatment_df["treatment_count"], errors="coerce").fillna(0).astype(int)
        return treatment_df

    def _build_treatment_lookup(self, treatment_df: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
        if treatment_df.empty:
            return {}

        lookup: dict[int, list[dict[str, Any]]] = {}
        for row in treatment_df.to_dict(orient="records"):
            hadm_id = int(row["hadm_id"])
            lookup.setdefault(hadm_id, []).append(row)
        return lookup

    def _build_feature_matrix(self, dataframe: pd.DataFrame) -> sparse.csr_matrix:
        numeric_df = dataframe[self.numeric_features].copy()
        for col in self.numeric_features:
            median = numeric_df[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            self._numeric_medians[col] = float(median)
            numeric_df[col] = numeric_df[col].fillna(median)

        self._numeric_scaler = StandardScaler()
        numeric_matrix = self._numeric_scaler.fit_transform(numeric_df.astype(float))
        numeric_sparse = sparse.csr_matrix(numeric_matrix)

        categorical_df = dataframe[["gender", "race"]].copy()
        categorical_df["gender"] = categorical_df["gender"].fillna("unknown")
        categorical_df["race"] = categorical_df["race"].fillna("unknown")

        self._gender_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        gender_matrix = self._gender_encoder.fit_transform(categorical_df[["gender"]])

        self._race_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        race_matrix = self._race_encoder.fit_transform(categorical_df[["race"]])

        self._icd_encoder = MultiLabelBinarizer(sparse_output=True)
        icd_matrix = self._icd_encoder.fit_transform(dataframe["diagnoses_icd_list"])

        self._icd_group_encoder = MultiLabelBinarizer(sparse_output=True)
        icd_group_matrix = self._icd_group_encoder.fit_transform(dataframe["diagnoses_3digit_list"])

        combined = sparse.hstack(
            [numeric_sparse, gender_matrix, race_matrix, icd_group_matrix, icd_matrix],
            format="csr",
        )

        if np.isnan(combined.data).any():
            raise ValueError("Feature matrix contains NaN values after preprocessing.")

        return combined

    def _transform_rows(self, dataframe: pd.DataFrame) -> sparse.csr_matrix:
        assert self._numeric_scaler is not None
        assert self._gender_encoder is not None
        assert self._race_encoder is not None
        assert self._icd_encoder is not None
        assert self._icd_group_encoder is not None

        numeric_df = dataframe[self.numeric_features].copy()
        for col in self.numeric_features:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").fillna(self._numeric_medians[col])

        numeric_sparse = sparse.csr_matrix(self._numeric_scaler.transform(numeric_df.astype(float)))
        gender_matrix = self._gender_encoder.transform(dataframe[["gender"]].fillna("unknown"))
        race_matrix = self._race_encoder.transform(dataframe[["race"]].fillna("unknown"))
        icd_groups = dataframe["diagnoses_3digit_list"].apply(
            lambda items: self._filter_known_items(items, self._icd_group_encoder.classes_)
        )
        icd_codes = dataframe["diagnoses_icd_list"].apply(
            lambda items: self._filter_known_items(items, self._icd_encoder.classes_)
        )
        icd_group_matrix = self._icd_group_encoder.transform(icd_groups)
        icd_matrix = self._icd_encoder.transform(icd_codes)

        combined = sparse.hstack(
            [numeric_sparse, gender_matrix, race_matrix, icd_group_matrix, icd_matrix],
            format="csr",
        )

        if np.isnan(combined.data).any():
            raise ValueError("Transformed feature matrix contains NaN values.")

        return combined

    def _prepare_profile_row(self, patient_profile: dict[str, Any]) -> dict[str, Any]:
        diagnoses_icd_list = self._normalize_list(patient_profile.get("diagnoses_icd_list", []))
        diagnoses_3digit_list = self._normalize_list(patient_profile.get("diagnoses_3digit_list", []))
        if not diagnoses_3digit_list and diagnoses_icd_list:
            diagnoses_3digit_list = self._derive_icd_groups(diagnoses_icd_list)

        diagnosis_count = patient_profile.get("diagnosis_count")
        if diagnosis_count is None:
            diagnosis_count = len(diagnoses_icd_list)

        unique_icd_count = patient_profile.get("unique_icd_count")
        if unique_icd_count is None:
            unique_icd_count = len(set(diagnoses_icd_list))

        diversity_ratio = patient_profile.get("diagnosis_diversity_ratio")
        if diversity_ratio is None:
            diversity_ratio = (float(unique_icd_count) / float(diagnosis_count)) if diagnosis_count else 0.0

        return {
            "subject_id": int(patient_profile.get("subject_id", -1)),
            "hadm_id": int(patient_profile.get("hadm_id", -1)),
            "age": self._coerce_optional_float(patient_profile.get("age")),
            "gender": str(patient_profile.get("gender", "unknown") or "unknown"),
            "race": str(patient_profile.get("race", "unknown") or "unknown"),
            "bmi": self._coerce_optional_float(patient_profile.get("bmi")),
            "systolic_bp": self._coerce_optional_float(patient_profile.get("systolic_bp")),
            "diastolic_bp": self._coerce_optional_float(patient_profile.get("diastolic_bp")),
            "los_days": self._coerce_optional_float(patient_profile.get("los_days")),
            "mortality": int(patient_profile.get("mortality", 0) or 0),
            "discharge_location": str(patient_profile.get("discharge_location", "unknown") or "unknown"),
            "primary_diagnosis_icd": str(patient_profile.get("primary_diagnosis_icd", diagnoses_icd_list[0] if diagnoses_icd_list else "")),
            "primary_diagnosis_title": str(patient_profile.get("primary_diagnosis_title", "")),
            "primary_icd_3digit": str(patient_profile.get("primary_icd_3digit", diagnoses_3digit_list[0] if diagnoses_3digit_list else "")),
            "diagnosis_count": int(diagnosis_count),
            "unique_icd_count": int(unique_icd_count),
            "icd_version_mix": str(patient_profile.get("icd_version_mix", "unknown")),
            "diagnosis_diversity_ratio": float(diversity_ratio),
            "diagnoses_icd_list": diagnoses_icd_list,
            "diagnoses_3digit_list": diagnoses_3digit_list,
        }

    def _treatment_records_for_neighbor(
        self,
        neighbor_row: pd.Series,
        similarity: float,
        neighbor_index: int,
    ) -> list[dict[str, Any]]:
        hadm_id = int(neighbor_row["hadm_id"])
        los_days = self._safe_positive_los(neighbor_row["los_days"])
        treatments = self.treatments_by_hadm.get(hadm_id, [])
        records = []

        for treatment in treatments:
            records.append(
                {
                    "neighbor_index": neighbor_index,
                    "subject_id": int(neighbor_row["subject_id"]),
                    "hadm_id": hadm_id,
                    "los_days": los_days,
                    "similarity": similarity,
                    "treatment_source": treatment["treatment_source"],
                    "treatment_key": treatment["treatment_key"],
                    "treatment_name": treatment["treatment_name"],
                    "treatment_count": int(treatment["treatment_count"]),
                }
            )

        return records

    def _score_treatments(self, treatment_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not treatment_records:
            return []

        treatment_df = pd.DataFrame(treatment_records)
        aggregated = (
            treatment_df.groupby(["treatment_source", "treatment_key", "treatment_name"], dropna=False)
            .agg(
                frequency=("hadm_id", "nunique"),
                average_los=("los_days", "mean"),
                weighted_score=("similarity", lambda s: 0.0),
                supporting_neighbors=("neighbor_index", lambda s: sorted(set(int(v) for v in s))),
                event_count_sum=("treatment_count", "sum"),
            )
            .reset_index()
        )

        weighted_scores = []
        for (_, _, _), group in treatment_df.groupby(["treatment_source", "treatment_key", "treatment_name"], dropna=False):
            weighted_scores.append(float(np.sum(group["similarity"] / group["los_days"])))
        aggregated["weighted_score"] = weighted_scores

        aggregated["score"] = aggregated["frequency"] * (1.0 / aggregated["average_los"].clip(lower=EPSILON_LOS))
        aggregated = aggregated.sort_values(
            by=["weighted_score", "score", "frequency", "treatment_name"],
            ascending=[False, False, False, True],
        )

        recommendations = []
        for row in aggregated.itertuples(index=False):
            recommendations.append(
                {
                    "treatment_source": row.treatment_source,
                    "treatment_key": row.treatment_key,
                    "treatment_name": row.treatment_name,
                    "frequency": int(row.frequency),
                    "average_los": float(row.average_los),
                    "score": float(row.score),
                    "weighted_score": float(row.weighted_score),
                    "event_count_sum": int(row.event_count_sum),
                    "supporting_neighbor_indices": list(row.supporting_neighbors),
                }
            )

        return recommendations

    def _feature_similarity_summary(self, query_row: pd.Series, neighbor_row: pd.Series) -> dict[str, Any]:
        return {
            "same_gender": query_row["gender"] == neighbor_row["gender"],
            "same_race": query_row["race"] == neighbor_row["race"],
            "age_difference": self._safe_abs_diff(query_row["age"], neighbor_row["age"]),
            "bmi_difference": self._safe_abs_diff(query_row["bmi"], neighbor_row["bmi"]),
            "systolic_bp_difference": self._safe_abs_diff(query_row["systolic_bp"], neighbor_row["systolic_bp"]),
            "diastolic_bp_difference": self._safe_abs_diff(query_row["diastolic_bp"], neighbor_row["diastolic_bp"]),
            "shared_diagnosis_count": len(
                self._shared_items(query_row["diagnoses_icd_list"], neighbor_row["diagnoses_icd_list"])
            ),
        }

    def _ensure_fitted(self) -> None:
        if self.model is None or self.dataframe is None or self.feature_matrix is None:
            raise RuntimeError("The CDSS model is not fitted. Call fit() first.")

    def _validate_patient_index(self, patient_index: int) -> None:
        assert self.dataframe is not None
        if patient_index < 0 or patient_index >= len(self.dataframe):
            raise IndexError(f"patient_index {patient_index} is out of range for {len(self.dataframe)} admissions.")

    @staticmethod
    def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        result = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return bool(result and result[0])

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return [str(item) for item in value.tolist() if item is not None and str(item) != ""]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item is not None and str(item) != ""]
        if value == "":
            return []
        return [str(value)]

    @staticmethod
    def _shared_items(left: Any, right: Any) -> list[str]:
        left_set = set(SimilarityBasedCDSS._normalize_list(left))
        right_set = set(SimilarityBasedCDSS._normalize_list(right))
        return sorted(left_set & right_set)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _safe_abs_diff(left: Any, right: Any) -> float | None:
        if left is None or right is None or pd.isna(left) or pd.isna(right):
            return None
        return float(abs(float(left) - float(right)))

    @staticmethod
    def _safe_positive_los(value: Any) -> float:
        if value is None or pd.isna(value):
            return EPSILON_LOS
        return max(float(value), EPSILON_LOS)

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _derive_icd_groups(diagnoses_icd_list: list[str]) -> list[str]:
        groups = []
        for code in diagnoses_icd_list:
            stripped = str(code).strip()
            if not stripped:
                continue
            groups.append(f"UNK_{stripped[:3]}")
        return groups

    @staticmethod
    def _filter_known_items(items: Any, known_classes: Any) -> list[str]:
        known = set(str(item) for item in known_classes)
        return [item for item in SimilarityBasedCDSS._normalize_list(items) if item in known]


@lru_cache(maxsize=2)
def _cached_cdss(db_path: str, n_neighbors: int) -> SimilarityBasedCDSS:
    system = SimilarityBasedCDSS(db_path=db_path, n_neighbors=n_neighbors)
    system.fit()
    return system


def recommend_for_patient(
    patient_index: int,
    db_path: str | Path | None = None,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    top_k: int | None = None,
) -> dict[str, Any]:
    resolved_db_path = Path(db_path) if db_path else Path(__file__).resolve().parent / "mimic.db"
    cdss = _cached_cdss(str(resolved_db_path), int(n_neighbors))
    return cdss.recommend_for_patient(patient_index=patient_index, top_k=top_k)


def recommend_for_profile(
    patient_profile: dict[str, Any],
    db_path: str | Path | None = None,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    top_k: int | None = None,
) -> dict[str, Any]:
    resolved_db_path = Path(db_path) if db_path else Path(__file__).resolve().parent / "mimic.db"
    cdss = _cached_cdss(str(resolved_db_path), int(n_neighbors))
    return cdss.recommend_for_profile(patient_profile=patient_profile, top_k=top_k)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run similarity-based treatment recommendations for one admission index.")
    parser.add_argument("patient_index", nargs="?", type=int, help="Zero-based patient/admission index in the merged dataframe.")
    parser.add_argument("--db-path", type=Path, default=Path(__file__).resolve().parent / "mimic.db")
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--patient-json",
        type=str,
        default=None,
        help="JSON string with custom patient details, for example: '{\"age\":65,\"gender\":\"F\",\"race\":\"WHITE\",\"bmi\":26,\"systolic_bp\":120,\"diastolic_bp\":80,\"diagnoses_icd_list\":[\"I10\",\"E119\"]}'",
    )
    args = parser.parse_args()

    if args.patient_json:
        result = recommend_for_profile(
            patient_profile=json.loads(args.patient_json),
            db_path=args.db_path,
            n_neighbors=args.neighbors,
            top_k=args.top_k,
        )
    else:
        if args.patient_index is None:
            parser.error("patient_index is required unless --patient-json is provided.")
        result = recommend_for_patient(
            patient_index=args.patient_index,
            db_path=args.db_path,
            n_neighbors=args.neighbors,
            top_k=args.top_k,
        )
    print(json.dumps(result, indent=2, default=_json_default))
