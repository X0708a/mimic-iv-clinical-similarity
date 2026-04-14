from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


TABLE_NAME = "cdss_diagnoses"
LIST_COLUMNS = ("diagnoses_icd_list", "diagnoses_3digit_list")
CRITICAL_NON_NULL_COLUMNS = (
    "hadm_id",
    "subject_id",
    "diagnosis_count",
    "unique_icd_count",
    "diagnoses_icd_list",
    "diagnoses_3digit_list",
    "icd_version_mix",
)
SELECT_COLUMNS = (
    "hadm_id",
    "subject_id",
    "primary_diagnosis_icd",
    "primary_icd_3digit",
    "diagnosis_count",
    "unique_icd_count",
    "diagnosis_diversity_ratio",
    "diagnoses_icd_list",
    "diagnoses_3digit_list",
    "icd_version_mix",
)
DEFAULT_DB_CANDIDATES = (
    Path("AI CDSS/mimic.db"),
    Path("mimic.db"),
    Path("mimic-iv-3.1/mimic.db"),
)


@dataclass(frozen=True)
class ValidationSummary:
    row_count: int
    list_columns_are_python_lists: bool
    critical_null_counts: dict[str, int]
    invalid_count_order: int

    @property
    def is_valid(self) -> bool:
        return (
            self.list_columns_are_python_lists
            and self.invalid_count_order == 0
            and all(count == 0 for count in self.critical_null_counts.values())
        )


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        candidate = Path(db_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"DuckDB file not found: {candidate}")

    search_roots = []
    for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
        resolved_root = root.resolve()
        if resolved_root not in search_roots:
            search_roots.append(resolved_root)

    checked_paths: list[str] = []
    for root in search_roots:
        for candidate in DEFAULT_DB_CANDIDATES:
            path = (root / candidate).resolve()
            checked_paths.append(str(path))
            if path.exists():
                return path

    checked = "\n".join(checked_paths)
    raise FileNotFoundError(f"Could not find mimic.db. Checked:\n{checked}")


def connect_duckdb(db_path: str | Path | None = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(resolve_db_path(db_path)), read_only=read_only)


def normalize_code_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return []
    raise TypeError(f"Unsupported diagnosis list value: {type(value)!r}")


def load_cdss_diagnoses(
    db_path: str | Path | None = None,
    validate: bool = True,
    prepare_sets: bool = True,
) -> pd.DataFrame:
    query = f"SELECT {', '.join(SELECT_COLUMNS)} FROM {TABLE_NAME}"
    con = connect_duckdb(db_path=db_path, read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    for column in LIST_COLUMNS:
        df[column] = df[column].map(normalize_code_list)

    if not df["hadm_id"].is_unique:
        duplicate_count = int(df["hadm_id"].duplicated().sum())
        raise ValueError(f"hadm_id must be unique. Found {duplicate_count} duplicate rows.")

    df = df.set_index("hadm_id", drop=False)

    if validate:
        summary = validate_cdss_diagnoses(df)
        if not summary.is_valid:
            raise ValueError(f"Validation failed: {summary}")

    if prepare_sets:
        prepare_similarity_columns(df)

    return df


def prepare_similarity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "dx_set" not in df.columns:
        df["dx_set"] = df["diagnoses_icd_list"].map(set)
    if "dx_3_set" not in df.columns:
        df["dx_3_set"] = df["diagnoses_3digit_list"].map(set)
    return df


def validate_cdss_diagnoses(df: pd.DataFrame) -> ValidationSummary:
    list_columns_are_python_lists = all(df[column].map(lambda value: isinstance(value, list)).all() for column in LIST_COLUMNS)
    critical_null_counts = {column: int(df[column].isna().sum()) for column in CRITICAL_NON_NULL_COLUMNS}
    invalid_count_order = int((df["diagnosis_count"] < df["unique_icd_count"]).sum())

    return ValidationSummary(
        row_count=int(len(df)),
        list_columns_are_python_lists=bool(list_columns_are_python_lists),
        critical_null_counts=critical_null_counts,
        invalid_count_order=invalid_count_order,
    )
