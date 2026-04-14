from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.clinical_similarity import (
    find_clinically_similar_patients,
    load_clinical_similarity_dataset,
    recommend_treatments,
)
from src.treatment_loader import load_cdss_treatment_features, validate_cdss_treatment_features
from src.treatment_similarity import baseline_treatment_similarity, find_similar_treatment_patients


def build_treatment_test_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))

    base_df = pd.DataFrame(
        [
            {"hadm_id": 201, "subject_id": 1, "age": 70, "gender": "F", "race": "WHITE", "los_days": 5.0, "mortality": 0},
            {"hadm_id": 202, "subject_id": 2, "age": 69, "gender": "F", "race": "WHITE", "los_days": 6.0, "mortality": 0},
            {"hadm_id": 203, "subject_id": 3, "age": 52, "gender": "M", "race": "BLACK", "los_days": 9.0, "mortality": 1},
            {"hadm_id": 204, "subject_id": 4, "age": 51, "gender": "M", "race": "BLACK", "los_days": 3.0, "mortality": 0},
        ]
    )
    con.register("base_df", base_df)
    con.execute("CREATE TABLE cdss_base AS SELECT * FROM base_df")

    prescriptions_df = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 201, "starttime": "2020-01-01 08:00:00", "stoptime": "2020-01-03 08:00:00", "drug_type": "MAIN", "drug": "Furosemide", "formulary_drug_cd": "FURO", "prod_strength": "40mg", "doses_per_24_hrs": 1, "route": "PO"},
            {"subject_id": 1, "hadm_id": 201, "starttime": "2020-01-01 08:00:00", "stoptime": "2020-01-02 08:00:00", "drug_type": "MAIN", "drug": "Aspirin", "formulary_drug_cd": "ASA", "prod_strength": "81mg", "doses_per_24_hrs": 1, "route": "PO"},
            {"subject_id": 2, "hadm_id": 202, "starttime": "2020-01-02 08:00:00", "stoptime": "2020-01-04 08:00:00", "drug_type": "MAIN", "drug": "Furosemide", "formulary_drug_cd": "FURO", "prod_strength": "40mg", "doses_per_24_hrs": 1, "route": "PO"},
            {"subject_id": 2, "hadm_id": 202, "starttime": "2020-01-02 08:00:00", "stoptime": "2020-01-03 08:00:00", "drug_type": "MAIN", "drug": "Aspirin", "formulary_drug_cd": "ASA", "prod_strength": "81mg", "doses_per_24_hrs": 1, "route": "PO"},
            {"subject_id": 3, "hadm_id": 203, "starttime": "2020-01-05 08:00:00", "stoptime": "2020-01-10 08:00:00", "drug_type": "MAIN", "drug": "Ceftriaxone", "formulary_drug_cd": "CEFT", "prod_strength": "1g", "doses_per_24_hrs": 1, "route": "IV"},
            {"subject_id": 4, "hadm_id": 204, "starttime": "2020-01-06 08:00:00", "stoptime": "2020-01-07 08:00:00", "drug_type": "MAIN", "drug": "Metformin", "formulary_drug_cd": "METF", "prod_strength": "500mg", "doses_per_24_hrs": 2, "route": "PO"},
        ]
    )
    con.register("prescriptions_df", prescriptions_df)
    con.execute("CREATE TABLE prescriptions AS SELECT * FROM prescriptions_df")

    procedures_df = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 201, "seq_num": 1, "chartdate": "2020-01-01", "icd_code": "5491", "icd_version": 9},
            {"subject_id": 2, "hadm_id": 202, "seq_num": 1, "chartdate": "2020-01-02", "icd_code": "5491", "icd_version": 9},
            {"subject_id": 3, "hadm_id": 203, "seq_num": 1, "chartdate": "2020-01-05", "icd_code": "0FT44ZZ", "icd_version": 10},
        ]
    )
    con.register("procedures_df", procedures_df)
    con.execute("CREATE TABLE procedures_icd AS SELECT * FROM procedures_df")

    d_procedures_df = pd.DataFrame(
        [
            {"icd_code": "5491", "icd_version": 9, "long_title": "Percutaneous abdominal drainage"},
            {"icd_code": "0FT44ZZ", "icd_version": 10, "long_title": "Resection of Gallbladder"},
        ]
    )
    con.register("d_procedures_df", d_procedures_df)
    con.execute("CREATE TABLE d_icd_procedures AS SELECT * FROM d_procedures_df")

    diagnoses_df = pd.DataFrame(
        [
            {
                "hadm_id": 201,
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
                "hadm_id": 202,
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
                "hadm_id": 203,
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
        ]
    )
    con.register("diagnoses_df", diagnoses_df)
    con.execute("CREATE TABLE cdss_diagnoses AS SELECT * FROM diagnoses_df")
    con.close()


def test_load_cdss_treatment_features_from_raw_tables(tmp_path: Path):
    db_path = tmp_path / "treatment.duckdb"
    build_treatment_test_db(db_path)

    df = load_cdss_treatment_features(db_path=db_path, include_raw_columns=False)
    summary = validate_cdss_treatment_features(df)

    assert summary.is_valid
    assert len(df) == 3
    assert isinstance(df.loc[201, "rx_drug_list"], list)
    assert "furosemide" in df.loc[201, "rx_drug_list"]
    assert df.loc[201, "proc_icd_list"] == ["549"]
    assert 0.0 <= df.loc[201, "treatment_complexity_score"] <= 1.0


def test_find_similar_treatment_patients_returns_expected_match(tmp_path: Path):
    db_path = tmp_path / "treatment.duckdb"
    build_treatment_test_db(db_path)
    df = load_cdss_treatment_features(db_path=db_path)

    results = find_similar_treatment_patients(df=df, query_hadm_id=201, k=2)

    assert results.iloc[0]["hadm_id"] == 202
    assert results.iloc[0]["composite_similarity"] >= results.iloc[1]["composite_similarity"]
    assert baseline_treatment_similarity(df.loc[201], df.loc[202]) > baseline_treatment_similarity(df.loc[201], df.loc[203])


def test_clinical_similarity_and_recommendations(tmp_path: Path):
    db_path = tmp_path / "treatment.duckdb"
    build_treatment_test_db(db_path)
    df = load_clinical_similarity_dataset(db_path=db_path)

    similar = find_clinically_similar_patients(df=df, query_hadm_id=201, k=2)
    recommendations = recommend_treatments(df=df, query_hadm_id=201, k=2, top_n=5)

    assert similar.iloc[0]["hadm_id"] == 202
    assert "furosemide" in set(recommendations["drug"])
