from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.ml_dataset import (
    build_phase3_feature_frame,
    load_joined_cdss_dataset,
    pick_outcome_column,
)
from src.ml_workflow import evaluate_similarity_outcome_alignment, run_phase1_validation


def build_ml_test_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))

    base_df = pd.DataFrame(
        [
            {
                "hadm_id": 101,
                "subject_id": 1,
                "age": 70,
                "gender": "F",
                "race": "WHITE",
                "los_days": 4.0,
                "mortality": 0,
                "readmission_30day": 0,
                "admission_type": "EMERGENCY",
            },
            {
                "hadm_id": 102,
                "subject_id": 2,
                "age": 71,
                "gender": "F",
                "race": "WHITE",
                "los_days": 5.0,
                "mortality": 0,
                "readmission_30day": 0,
                "admission_type": "EMERGENCY",
            },
            {
                "hadm_id": 103,
                "subject_id": 3,
                "age": 45,
                "gender": "M",
                "race": "BLACK",
                "los_days": 9.0,
                "mortality": 1,
                "readmission_30day": 1,
                "admission_type": "URGENT",
            },
            {
                "hadm_id": 104,
                "subject_id": 4,
                "age": 46,
                "gender": "M",
                "race": "BLACK",
                "los_days": 8.0,
                "mortality": 1,
                "readmission_30day": 1,
                "admission_type": "URGENT",
            },
        ]
    )
    con.register("base_df", base_df)
    con.execute("CREATE TABLE cdss_base AS SELECT * FROM base_df")

    diagnoses_df = pd.DataFrame(
        [
            {
                "hadm_id": 101,
                "subject_id": 1,
                "primary_diagnosis_icd": "I10",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 102,
                "subject_id": 2,
                "primary_diagnosis_icd": "I10",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 103,
                "subject_id": 3,
                "primary_diagnosis_icd": "J189",
                "primary_icd_3digit": "10_J18",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["J189", "R509"],
                "diagnoses_3digit_list": ["10_J18", "10_R50"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 104,
                "subject_id": 4,
                "primary_diagnosis_icd": "J189",
                "primary_icd_3digit": "10_J18",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["J189", "R509"],
                "diagnoses_3digit_list": ["10_J18", "10_R50"],
                "icd_version_mix": "ICD10_only",
            },
        ]
    )
    con.register("diagnoses_df", diagnoses_df)
    con.execute("CREATE TABLE cdss_diagnoses AS SELECT * FROM diagnoses_df")

    treatment_df = pd.DataFrame(
        [
            {
                "hadm_id": 101,
                "subject_id": 1,
                "treatment_source": "prescription",
                "treatment_key": "FUROSEMIDE",
                "treatment_name": "Furosemide",
                "treatment_count": 1,
                "first_treatment_time": "2020-01-01 00:00:00",
                "last_treatment_time": "2020-01-01 00:00:00",
            },
            {
                "hadm_id": 103,
                "subject_id": 3,
                "treatment_source": "prescription",
                "treatment_key": "CEFTRIAXONE",
                "treatment_name": "Ceftriaxone",
                "treatment_count": 2,
                "first_treatment_time": "2020-01-01 00:00:00",
                "last_treatment_time": "2020-01-02 00:00:00",
            },
        ]
    )
    con.register("treatment_df", treatment_df)
    con.execute("CREATE TABLE cdss_treatment AS SELECT * FROM treatment_df")
    con.close()


def test_load_joined_cdss_dataset_adds_treatment_summary(tmp_path: Path):
    db_path = tmp_path / "ml.duckdb"
    build_ml_test_db(db_path)

    df = load_joined_cdss_dataset(db_path=db_path)

    assert len(df) == 4
    assert df.loc[101, "treatment_event_count"] == 1
    assert df.loc[102, "treatment_event_count"] == 0
    assert isinstance(df.loc[101, "dx_set"], set)


def test_pick_outcome_column_prefers_mortality(tmp_path: Path):
    db_path = tmp_path / "ml.duckdb"
    build_ml_test_db(db_path)
    df = load_joined_cdss_dataset(db_path=db_path)

    assert pick_outcome_column(df) == "mortality"
    assert pick_outcome_column(df, preferred="readmission_30day") == "readmission_30day"


def test_build_phase3_feature_frame(tmp_path: Path):
    db_path = tmp_path / "ml.duckdb"
    build_ml_test_db(db_path)
    df = load_joined_cdss_dataset(db_path=db_path)

    features = build_phase3_feature_frame(df)

    assert "age" in features.columns
    assert "elderly" in features.columns
    assert "diagnosis_count" in features.columns
    assert "treatment_event_count" in features.columns
    assert features.loc[101, "elderly"] == 1


def test_run_phase1_validation_returns_summary(tmp_path: Path):
    db_path = tmp_path / "ml.duckdb"
    build_ml_test_db(db_path)

    df, summary = run_phase1_validation(db_path=db_path)

    assert len(df) == 4
    assert summary.row_count == 4
    assert summary.list_columns_are_python_lists
    assert summary.quick_similarity_max == 1.0


def test_evaluate_similarity_outcome_alignment(tmp_path: Path):
    db_path = tmp_path / "ml.duckdb"
    build_ml_test_db(db_path)
    df = load_joined_cdss_dataset(db_path=db_path)

    summary, band_summary = evaluate_similarity_outcome_alignment(
        df=df,
        outcome_column="mortality",
        sample_size=4,
        random_state=42,
        bins=3,
    )

    assert summary.outcome_column == "mortality"
    assert summary.sampled_patients == 4
    assert summary.pair_count == 6
    assert not band_summary.empty
