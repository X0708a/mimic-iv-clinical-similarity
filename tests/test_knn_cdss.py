from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "AI CDSS" / "knn_cdss.py"


def load_module():
    spec = importlib.util.spec_from_file_location("knn_cdss", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_test_db(db_path: Path, with_cdss_treatment: bool = True) -> None:
    con = duckdb.connect(str(db_path))

    base_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 101,
                "age": 65,
                "gender": "F",
                "race": "WHITE",
                "bmi": 26.0,
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "los_days": 4.0,
                "mortality": 0,
                "discharge_location": "HOME",
            },
            {
                "subject_id": 2,
                "hadm_id": 102,
                "age": 66,
                "gender": "F",
                "race": "WHITE",
                "bmi": 25.5,
                "systolic_bp": 118.0,
                "diastolic_bp": 78.0,
                "los_days": 3.0,
                "mortality": 0,
                "discharge_location": "HOME",
            },
            {
                "subject_id": 3,
                "hadm_id": 103,
                "age": 70,
                "gender": "M",
                "race": "BLACK",
                "bmi": None,
                "systolic_bp": None,
                "diastolic_bp": 76.0,
                "los_days": 6.0,
                "mortality": 1,
                "discharge_location": "SNF",
            },
        ]
    )
    con.register("base_df", base_df)
    con.execute("CREATE TABLE cdss_base AS SELECT * FROM base_df")

    diagnoses_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 101,
                "primary_diagnosis_icd": "I10",
                "primary_diagnosis_title": "Hypertension",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "icd_version_mix": "ICD10_only",
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
            },
            {
                "subject_id": 2,
                "hadm_id": 102,
                "primary_diagnosis_icd": "I10",
                "primary_diagnosis_title": "Hypertension",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "icd_version_mix": "ICD10_only",
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
            },
            {
                "subject_id": 3,
                "hadm_id": 103,
                "primary_diagnosis_icd": "J189",
                "primary_diagnosis_title": "Pneumonia",
                "primary_icd_3digit": "10_J18",
                "diagnosis_count": 1,
                "unique_icd_count": 1,
                "icd_version_mix": "ICD10_only",
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["J189"],
                "diagnoses_3digit_list": ["10_J18"],
            },
        ]
    )
    con.register("diagnoses_df", diagnoses_df)
    con.execute("CREATE TABLE cdss_diagnoses AS SELECT * FROM diagnoses_df")

    if with_cdss_treatment:
        treatment_df = pd.DataFrame(
            [
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "treatment_source": "prescription",
                    "treatment_key": "FUROSEMIDE",
                    "treatment_name": "Furosemide",
                    "treatment_count": 1,
                },
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "treatment_source": "procedure",
                    "treatment_key": "10:5A1955Z",
                    "treatment_name": "Respiratory Ventilation",
                    "treatment_count": 1,
                },
                {
                    "subject_id": 3,
                    "hadm_id": 103,
                    "treatment_source": "prescription",
                    "treatment_key": "CEFTRIAXONE",
                    "treatment_name": "Ceftriaxone",
                    "treatment_count": 1,
                },
            ]
        )
        con.register("treatment_df", treatment_df)
        con.execute("CREATE TABLE cdss_treatment AS SELECT * FROM treatment_df")
    else:
        prescriptions_df = pd.DataFrame(
            [
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "drug": "Furosemide",
                },
                {
                    "subject_id": 3,
                    "hadm_id": 103,
                    "drug": "Ceftriaxone",
                },
            ]
        )
        procedures_df = pd.DataFrame(
            [
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "icd_code": "5A1955Z",
                    "icd_version": 10,
                }
            ]
        )
        d_procedures_df = pd.DataFrame(
            [
                {
                    "icd_code": "5A1955Z",
                    "icd_version": 10,
                    "long_title": "Respiratory Ventilation",
                }
            ]
        )
        con.register("prescriptions_df", prescriptions_df)
        con.register("procedures_df", procedures_df)
        con.register("d_procedures_df", d_procedures_df)
        con.execute("CREATE TABLE prescriptions AS SELECT * FROM prescriptions_df")
        con.execute("CREATE TABLE procedures_icd AS SELECT * FROM procedures_df")
        con.execute("CREATE TABLE d_icd_procedures AS SELECT * FROM d_procedures_df")

    con.close()


def test_recommend_for_patient_with_cdss_treatment(tmp_path: Path):
    module = load_module()
    db_path = tmp_path / "test_cdss.duckdb"
    build_test_db(db_path, with_cdss_treatment=True)

    system = module.SimilarityBasedCDSS(db_path=db_path, n_neighbors=2).fit()
    result = system.recommend_for_patient(0, top_k=2)

    assert len(result["similar_patients"]) == 2
    assert result["similar_patients"][0]["hadm_id"] == 102
    assert result["similar_patients"][0]["similarity"] >= result["similar_patients"][1]["similarity"]
    assert result["recommended_treatments"][0]["treatment_name"] == "Furosemide"
    assert result["recommended_treatments"][0]["average_los"] > 0
    assert result["recommended_treatments"][0]["weighted_score"] > 0


def test_recommend_for_patient_falls_back_to_raw_treatments(tmp_path: Path):
    module = load_module()
    db_path = tmp_path / "fallback_cdss.duckdb"
    build_test_db(db_path, with_cdss_treatment=False)

    result = module.recommend_for_patient(0, db_path=db_path, n_neighbors=2, top_k=2)

    treatment_names = [item["treatment_name"] for item in result["recommended_treatments"]]
    assert "Furosemide" in treatment_names
    assert "Respiratory Ventilation" in treatment_names


def test_feature_matrix_contains_no_nan_after_imputation(tmp_path: Path):
    module = load_module()
    db_path = tmp_path / "nan_check_cdss.duckdb"
    build_test_db(db_path, with_cdss_treatment=True)

    system = module.SimilarityBasedCDSS(db_path=db_path, n_neighbors=2).fit()
    assert system.feature_matrix is not None
    assert not pd.isna(system.feature_matrix.data).any()


def test_recommend_for_custom_profile_returns_similarity_scores(tmp_path: Path):
    module = load_module()
    db_path = tmp_path / "profile_cdss.duckdb"
    build_test_db(db_path, with_cdss_treatment=True)

    result = module.recommend_for_profile(
        {
            "age": 66,
            "gender": "F",
            "race": "WHITE",
            "bmi": 25.7,
            "systolic_bp": 119,
            "diastolic_bp": 79,
            "diagnoses_icd_list": ["I10", "E119"],
        },
        db_path=db_path,
        n_neighbors=2,
        top_k=2,
    )

    assert len(result["similar_patients"]) == 2
    assert result["similar_patients"][0]["hadm_id"] == 102
    assert 0.0 <= result["similar_patients"][0]["similarity"] <= 1.0
    assert "Furosemide" in [item["treatment_name"] for item in result["recommended_treatments"]]
