from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.model_benchmark import (
    _fit_benchmark_models,
    TopKJaccardVotingClassifier,
    run_model_diagnostics,
    load_benchmark_dataset,
    pick_binary_target_column,
    run_model_benchmark,
    save_residual_plot,
    split_benchmark_dataset,
)


def build_benchmark_test_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))

    base_rows: list[dict[str, object]] = []
    diagnosis_rows: list[dict[str, object]] = []
    treatment_rows: list[dict[str, object]] = []

    for offset in range(6):
        hadm_id = 1001 + offset
        subject_id = 501 + offset
        base_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "age": 68 + offset,
                "gender": "F",
                "race": "WHITE",
                "admission_type": "EMERGENCY",
                "admission_location": "ER",
                "discharge_location": "HOME",
                "insurance": "Medicare",
                "language": "ENGLISH",
                "los_days": 3.0 + (offset * 0.2),
                "mortality": 0,
                "bmi": 27.0 + (offset * 0.1),
                "systolic_bp": 128.0 + offset,
                "diastolic_bp": 78.0 + offset,
            }
        )
        diagnosis_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "primary_diagnosis_icd": "I10",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 3,
                "unique_icd_count": 3,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119", "N189"],
                "diagnoses_3digit_list": ["10_I10", "10_E11", "10_N18"],
                "icd_version_mix": "ICD10_only",
            }
        )
        treatment_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "procedure_list": ["3991"],
                "proc_icd_list": ["399"],
                "proc_icd_versions": "10",
                "proc_count": 1,
                "proc_group_count": 1,
                "surgery_count": 0,
                "rx_drug_list": ["lisinopril", "metformin"],
                "rx_routes_list": ["PO"],
                "route_diversity": 1,
                "rx_drug_types": ["MAIN"],
                "rx_total_count": 2,
                "rx_unique_drugs": 2,
                "rx_unique_formulary_drugs": 2,
                "rx_iv_count": 0,
                "rx_oral_count": 2,
                "rx_iv_ratio": 0.0,
                "treatment_days": 2,
                "avg_rx_duration_days": 1.5,
                "treatment_complexity_score": 0.15 + (offset * 0.01),
                "treatment_intensity_label": "Low",
            }
        )

    for offset in range(6):
        hadm_id = 2001 + offset
        subject_id = 601 + offset
        base_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "age": 49 + offset,
                "gender": "M",
                "race": "BLACK",
                "admission_type": "URGENT",
                "admission_location": "CLINIC",
                "discharge_location": "SNF",
                "insurance": "Medicaid",
                "language": "ENGLISH",
                "los_days": 8.0 + (offset * 0.3),
                "mortality": 1,
                "bmi": 31.0 + (offset * 0.2),
                "systolic_bp": 95.0 - offset,
                "diastolic_bp": 58.0 - offset,
            }
        )
        diagnosis_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "primary_diagnosis_icd": "J189",
                "primary_icd_3digit": "10_J18",
                "diagnosis_count": 3,
                "unique_icd_count": 3,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["J189", "R509", "A419"],
                "diagnoses_3digit_list": ["10_J18", "10_R50", "10_A41"],
                "icd_version_mix": "ICD10_only",
            }
        )
        treatment_rows.append(
            {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "procedure_list": ["9671"],
                "proc_icd_list": ["967"],
                "proc_icd_versions": "10",
                "proc_count": 1,
                "proc_group_count": 1,
                "surgery_count": 1,
                "rx_drug_list": ["ceftriaxone", "azithromycin"],
                "rx_routes_list": ["IV"],
                "route_diversity": 1,
                "rx_drug_types": ["MAIN"],
                "rx_total_count": 2,
                "rx_unique_drugs": 2,
                "rx_unique_formulary_drugs": 2,
                "rx_iv_count": 2,
                "rx_oral_count": 0,
                "rx_iv_ratio": 1.0,
                "treatment_days": 6,
                "avg_rx_duration_days": 4.0,
                "treatment_complexity_score": 0.85 - (offset * 0.01),
                "treatment_intensity_label": "High",
            }
        )

    con.register("base_df", pd.DataFrame(base_rows))
    con.execute("CREATE TABLE cdss_base AS SELECT * FROM base_df")
    con.register("diagnosis_df", pd.DataFrame(diagnosis_rows))
    con.execute("CREATE TABLE cdss_diagnoses AS SELECT * FROM diagnosis_df")
    con.register("treatment_df", pd.DataFrame(treatment_rows))
    con.execute("CREATE TABLE cdss_treatment AS SELECT * FROM treatment_df")
    con.close()


def test_pick_binary_target_column_prefers_mortality(tmp_path: Path):
    db_path = tmp_path / "benchmark.duckdb"
    build_benchmark_test_db(db_path)
    df = load_benchmark_dataset(db_path)

    assert pick_binary_target_column(df) == "mortality"


def test_topk_jaccard_voting_classifier_predicts_cluster_labels(tmp_path: Path):
    db_path = tmp_path / "benchmark.duckdb"
    build_benchmark_test_db(db_path)
    df = load_benchmark_dataset(db_path)
    train_df, test_df, y_train, y_test = split_benchmark_dataset(df, target_column="mortality", random_state=42)

    model = TopKJaccardVotingClassifier(top_k=3, candidate_limit=50)
    model.fit(train_df, y_train.to_numpy())

    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)

    assert len(predictions) == len(test_df)
    assert probabilities.shape == (len(test_df), 2)
    assert (predictions == y_test.to_numpy()).mean() >= 0.66


def test_run_model_benchmark_returns_comparison_table(tmp_path: Path):
    db_path = tmp_path / "benchmark.duckdb"
    build_benchmark_test_db(db_path)

    comparison_df, summary, analysis = run_model_benchmark(
        db_path=db_path,
        target_column="mortality",
        random_state=42,
        knn_neighbors=3,
        custom_top_k=3,
        custom_candidate_limit=50,
        include_xgboost=False,
    )

    assert set(comparison_df["model"]) == {
        "Logistic Regression",
        "Random Forest",
        "kNN (cosine)",
        "Custom Jaccard KNN",
    }
    assert {"accuracy", "precision", "recall", "f1_score", "roc_auc"}.issubset(comparison_df.columns)
    assert summary.target_column == "mortality"
    assert summary.row_count == 12
    assert summary.train_size + summary.test_size == 12
    assert summary.best_model_name in set(comparison_df["model"])
    assert "Best model on mortality" in analysis


def test_run_model_diagnostics_returns_confusion_matrix_and_importance(tmp_path: Path):
    db_path = tmp_path / "benchmark.duckdb"
    build_benchmark_test_db(db_path)

    comparison_df, summary, analysis, confusion_df, feature_tables = run_model_diagnostics(
        db_path=db_path,
        target_column="mortality",
        random_state=42,
        knn_neighbors=3,
        custom_top_k=3,
        custom_candidate_limit=50,
        include_xgboost=False,
        top_n_features=5,
    )

    assert summary.target_column == "mortality"
    assert "Best model on mortality" in analysis
    assert len(confusion_df) == len(comparison_df)
    assert {"tn", "fp", "fn", "tp"}.issubset(confusion_df.columns)
    assert "Logistic Regression" in feature_tables
    assert "Random Forest" in feature_tables
    assert len(feature_tables["Logistic Regression"]) <= 5


def test_save_residual_plot_writes_png(tmp_path: Path):
    db_path = tmp_path / "benchmark.duckdb"
    build_benchmark_test_db(db_path)

    artifacts = _fit_benchmark_models(
        db_path=db_path,
        target_column="mortality",
        random_state=42,
        knn_neighbors=3,
        custom_top_k=3,
        custom_candidate_limit=50,
        include_xgboost=False,
    )
    output_path = tmp_path / "residual_plot.png"
    saved_path = save_residual_plot(artifacts=artifacts, output_path=output_path, model_name="Logistic Regression")

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
