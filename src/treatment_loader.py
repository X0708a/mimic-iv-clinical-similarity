from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import connect_duckdb, normalize_code_list, resolve_db_path
except ImportError:  # pragma: no cover
    from data_loader import connect_duckdb, normalize_code_list, resolve_db_path


AGGREGATED_TREATMENT_REQUIRED_COLUMNS = (
    "hadm_id",
    "subject_id",
    "rx_drug_list",
    "proc_icd_list",
    "treatment_complexity_score",
    "treatment_intensity_label",
)
TREATMENT_LIST_COLUMNS = (
    "rx_drug_list",
    "proc_icd_list",
    "procedure_list",
    "rx_routes_list",
    "rx_drug_types",
    "proc_version_list",
)
RAW_LIST_COLUMNS = ("procedures_icd9", "procedures_icd10", "prescriptions_raw")
TREATMENT_CRITICAL_COLUMNS = (
    "hadm_id",
    "subject_id",
    "rx_drug_list",
    "proc_icd_list",
    "rx_unique_drugs",
    "proc_count",
    "treatment_days",
    "treatment_complexity_score",
    "treatment_intensity_label",
)
DEFAULT_TREATMENT_COLUMNS = (
    "subject_id",
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
)


@dataclass(frozen=True)
class TreatmentValidationSummary:
    row_count: int
    list_columns_are_python_lists: bool
    critical_null_counts: dict[str, int]
    invalid_count_relationships: int
    invalid_ratio_rows: int

    @property
    def is_valid(self) -> bool:
        return (
            self.list_columns_are_python_lists
            and self.invalid_count_relationships == 0
            and self.invalid_ratio_rows == 0
            and all(count == 0 for count in self.critical_null_counts.values())
        )


def _get_cdss_treatment_columns(db_path: str | Path | None = None) -> list[str]:
    con = connect_duckdb(db_path=db_path, read_only=True)
    try:
        return con.execute("DESCRIBE cdss_treatment").fetchdf()["column_name"].tolist()
    except Exception:
        return []
    finally:
        con.close()


def has_aggregated_cdss_treatment_schema(db_path: str | Path | None = None) -> bool:
    columns = set(_get_cdss_treatment_columns(db_path))
    return all(column in columns for column in AGGREGATED_TREATMENT_REQUIRED_COLUMNS)


def _build_raw_treatment_feature_query(include_raw_columns: bool = False) -> str:
    prescription_raw_select = ""
    procedure_raw_select = ""

    if include_raw_columns:
        prescription_raw_select = """
            ,
            list(
                struct_pack(
                    drug := drug_original,
                    drug_type := drug_type_original,
                    formulary_drug_cd := formulary_drug_cd,
                    prod_strength := prod_strength,
                    doses_per_24_hrs := doses_per_24_hrs,
                    route := route_original,
                    starttime := rx_start,
                    stoptime := rx_stop
                )
                ORDER BY rx_start, drug_name
            ) AS prescriptions_raw
        """
        procedure_raw_select = """
            ,
            list(
                struct_pack(
                    icd_code := icd_code,
                    long_title := long_title
                )
                ORDER BY charttime, icd_code
            ) FILTER (WHERE icd_version = 9) AS procedures_icd9
            ,
            list(
                struct_pack(
                    icd_code := icd_code,
                    long_title := long_title
                )
                ORDER BY charttime, icd_code
            ) FILTER (WHERE icd_version = 10) AS procedures_icd10
        """

    return f"""
    WITH rx_clean AS (
        SELECT
            p.subject_id,
            p.hadm_id,
            lower(trim(p.drug)) AS drug_name,
            trim(p.drug) AS drug_original,
            upper(trim(coalesce(p.route, 'UNKNOWN'))) AS route_name,
            trim(coalesce(p.route, 'UNKNOWN')) AS route_original,
            upper(trim(coalesce(p.drug_type, 'UNKNOWN'))) AS drug_type_name,
            trim(coalesce(p.drug_type, 'UNKNOWN')) AS drug_type_original,
            NULLIF(trim(coalesce(p.formulary_drug_cd, '')), '') AS formulary_drug_cd,
            p.prod_strength,
            p.doses_per_24_hrs,
            coalesce(try_cast(p.starttime AS TIMESTAMP), try_cast(p.stoptime AS TIMESTAMP)) AS rx_start,
            CASE
                WHEN p.starttime IS NOT NULL
                 AND p.stoptime IS NOT NULL
                 AND try_cast(p.stoptime AS TIMESTAMP) < try_cast(p.starttime AS TIMESTAMP) THEN try_cast(p.starttime AS TIMESTAMP)
                ELSE coalesce(try_cast(p.stoptime AS TIMESTAMP), try_cast(p.starttime AS TIMESTAMP))
            END AS rx_stop,
            CASE
                WHEN upper(trim(coalesce(p.route, ''))) LIKE 'IV%'
                  OR upper(trim(coalesce(p.route, ''))) IN ('INTRAVENOUS', 'IV DRIP', 'IVPCA', 'IVPB')
                  THEN 1
                ELSE 0
            END AS is_iv,
            CASE
                WHEN upper(trim(coalesce(p.route, ''))) LIKE 'PO%'
                  OR upper(trim(coalesce(p.route, ''))) IN ('PO', 'PO/NG', 'PO NG', 'PO-NG', 'ORAL', 'NG')
                  THEN 1
                ELSE 0
            END AS is_oral
        FROM prescriptions p
        INNER JOIN cdss_base b
            ON p.hadm_id = b.hadm_id
        WHERE p.hadm_id IS NOT NULL
          AND p.drug IS NOT NULL
          AND trim(p.drug) <> ''
          AND NOT (
              lower(trim(p.drug)) LIKE '%saline%'
              OR lower(trim(p.drug)) LIKE '%dextrose%'
              OR lower(trim(p.drug)) LIKE '%flush%'
          )
    ),
    rx_agg AS (
        SELECT
            any_value(subject_id) AS subject_id,
            hadm_id,
            list_sort(list(DISTINCT drug_name)) AS rx_drug_list,
            list_sort(list(DISTINCT route_name)) AS rx_routes_list,
            count(DISTINCT route_name) AS route_diversity,
            list_sort(list(DISTINCT drug_type_name)) AS rx_drug_types,
            count(*) AS rx_total_count,
            count(DISTINCT drug_name) AS rx_unique_drugs,
            count(DISTINCT formulary_drug_cd) AS rx_unique_formulary_drugs,
            sum(is_iv) AS rx_iv_count,
            sum(is_oral) AS rx_oral_count,
            cast(sum(is_iv) AS DOUBLE) / nullif(count(*), 0) AS rx_iv_ratio,
            coalesce(
                greatest(date_diff('day', min(cast(rx_start AS DATE)), max(cast(rx_stop AS DATE))) + 1, 1),
                1
            ) AS treatment_days,
            avg(
                CASE
                    WHEN rx_start IS NULL OR rx_stop IS NULL THEN 0.0
                    ELSE greatest(date_diff('second', rx_start, rx_stop), 0) / 86400.0
                END
            ) AS avg_rx_duration_days
            {prescription_raw_select}
        FROM rx_clean
        GROUP BY hadm_id
    ),
    proc_clean AS (
        SELECT
            p.subject_id,
            p.hadm_id,
            trim(p.icd_code) AS icd_code,
            cast(p.icd_version AS INTEGER) AS icd_version,
            cast(p.chartdate AS TIMESTAMP) AS charttime,
            substr(trim(p.icd_code), 1, 3) AS proc_group,
            coalesce(dx.long_title, trim(p.icd_code)) AS long_title,
            CASE
                WHEN p.icd_version = 9 THEN coalesce(try_cast(substr(trim(p.icd_code), 1, 2) AS INTEGER) BETWEEN 1 AND 86, false)
                WHEN p.icd_version = 10 THEN substr(trim(p.icd_code), 1, 1) = '0'
                ELSE false
            END AS is_surgery
        FROM procedures_icd p
        INNER JOIN cdss_base b
            ON p.hadm_id = b.hadm_id
        LEFT JOIN d_icd_procedures dx
            ON trim(p.icd_code) = trim(dx.icd_code)
           AND p.icd_version = dx.icd_version
        WHERE p.hadm_id IS NOT NULL
          AND p.icd_code IS NOT NULL
          AND trim(p.icd_code) <> ''
    ),
    proc_agg AS (
        SELECT
            any_value(subject_id) AS subject_id,
            hadm_id,
            list_sort(list(DISTINCT icd_code)) AS procedure_list,
            list_sort(list(DISTINCT proc_group)) AS proc_icd_list,
            list_sort(list(DISTINCT cast(icd_version AS VARCHAR))) AS proc_version_list,
            count(*) AS proc_count,
            count(DISTINCT proc_group) AS proc_group_count,
            sum(CASE WHEN is_surgery THEN 1 ELSE 0 END) AS surgery_count
            {procedure_raw_select}
        FROM proc_clean
        GROUP BY hadm_id
    )
    SELECT
        coalesce(rx.subject_id, proc.subject_id) AS subject_id,
        rx.hadm_id,
        proc.procedure_list,
        proc.proc_icd_list,
        proc.proc_version_list,
        proc.proc_count,
        proc.proc_group_count,
        proc.surgery_count,
        rx.rx_drug_list,
        rx.rx_routes_list,
        rx.route_diversity,
        rx.rx_drug_types,
        rx.rx_total_count,
        rx.rx_unique_drugs,
        rx.rx_unique_formulary_drugs,
        rx.rx_iv_count,
        rx.rx_oral_count,
        coalesce(rx.rx_iv_ratio, 0.0) AS rx_iv_ratio,
        rx.treatment_days,
        coalesce(rx.avg_rx_duration_days, 0.0) AS avg_rx_duration_days
        {", proc.procedures_icd9, proc.procedures_icd10, rx.prescriptions_raw" if include_raw_columns else ""}
    FROM rx_agg rx
    INNER JOIN proc_agg proc
        ON rx.hadm_id = proc.hadm_id
    """


def _normalize_existing_aggregated_frame(df: pd.DataFrame, include_raw_columns: bool = False) -> pd.DataFrame:
    list_columns = [column for column in (*TREATMENT_LIST_COLUMNS, *RAW_LIST_COLUMNS) if column in df.columns]
    for column in list_columns:
        df[column] = df[column].map(normalize_code_list)
    return df


def _compute_treatment_complexity(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_series(series: pd.Series) -> pd.Series:
        min_value = float(series.min())
        max_value = float(series.max())
        if max_value == min_value:
            fill_value = 0.0 if max_value == 0.0 else 1.0
            return pd.Series(fill_value, index=series.index, dtype=float)
        return (series.astype(float) - min_value) / (max_value - min_value)

    drug_diversity_norm = normalize_series(df["rx_unique_drugs"])
    procedure_count_norm = normalize_series(df["proc_count"])
    duration_norm = normalize_series(df["treatment_days"])
    iv_intensity_norm = df["rx_iv_ratio"].fillna(0.0).clip(lower=0.0, upper=1.0)

    df["treatment_complexity_score"] = (
        0.35 * drug_diversity_norm
        + 0.30 * procedure_count_norm
        + 0.20 * iv_intensity_norm
        + 0.15 * duration_norm
    ).clip(lower=0.0, upper=1.0)

    score_with_surgery_bonus = (df["treatment_complexity_score"] + np.where(df["surgery_count"] > 0, 0.15, 0.0)).clip(
        upper=1.0
    )
    labels = pd.cut(
        score_with_surgery_bonus,
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )
    df["treatment_intensity_label"] = labels.astype(str)
    return df


def prepare_treatment_similarity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "rx_set" not in df.columns:
        df["rx_set"] = df["rx_drug_list"].map(set)
    if "proc_set" not in df.columns:
        df["proc_set"] = df["proc_icd_list"].map(set)
    return df


def validate_cdss_treatment_features(df: pd.DataFrame) -> TreatmentValidationSummary:
    list_columns = [column for column in (*TREATMENT_LIST_COLUMNS, *RAW_LIST_COLUMNS) if column in df.columns]
    list_columns_are_python_lists = all(df[column].map(lambda value: isinstance(value, list)).all() for column in list_columns)
    critical_null_counts = {column: int(df[column].isna().sum()) for column in TREATMENT_CRITICAL_COLUMNS if column in df.columns}
    invalid_count_relationships = 0
    if {"rx_total_count", "rx_unique_drugs"}.issubset(df.columns):
        invalid_count_relationships += int((df["rx_total_count"] < df["rx_unique_drugs"]).sum())
    if {"proc_count", "proc_group_count"}.issubset(df.columns):
        invalid_count_relationships += int((df["proc_count"] < df["proc_group_count"]).sum())
    invalid_ratio_rows = int(((df["rx_iv_ratio"] < 0) | (df["rx_iv_ratio"] > 1)).sum()) if "rx_iv_ratio" in df.columns else 0

    return TreatmentValidationSummary(
        row_count=int(len(df)),
        list_columns_are_python_lists=bool(list_columns_are_python_lists),
        critical_null_counts=critical_null_counts,
        invalid_count_relationships=invalid_count_relationships,
        invalid_ratio_rows=invalid_ratio_rows,
    )


def load_cdss_treatment_features(
    db_path: str | Path | None = None,
    include_raw_columns: bool = False,
    validate: bool = True,
    prepare_sets: bool = True,
) -> pd.DataFrame:
    resolved_db_path = resolve_db_path(db_path)
    con = connect_duckdb(resolved_db_path, read_only=True)
    try:
        if has_aggregated_cdss_treatment_schema(resolved_db_path):
            available_columns = set(_get_cdss_treatment_columns(resolved_db_path))
            selected_columns = [column for column in DEFAULT_TREATMENT_COLUMNS if column in available_columns]
            if include_raw_columns:
                selected_columns += [column for column in RAW_LIST_COLUMNS if column in available_columns]
            query = f"SELECT {', '.join(selected_columns)} FROM cdss_treatment"
            df = con.execute(query).fetchdf()
            df = _normalize_existing_aggregated_frame(df, include_raw_columns=include_raw_columns)
        else:
            query = _build_raw_treatment_feature_query(include_raw_columns=include_raw_columns)
            df = con.execute(query).fetchdf()
            for column in [column for column in (*TREATMENT_LIST_COLUMNS, *RAW_LIST_COLUMNS) if column in df.columns]:
                df[column] = df[column].map(normalize_code_list)
            if "proc_version_list" in df.columns:
                df["proc_icd_versions"] = df["proc_version_list"].map(lambda values: ",".join(values))
                df = df.drop(columns=["proc_version_list"])
            df = _compute_treatment_complexity(df)
    finally:
        con.close()

    if not df["hadm_id"].is_unique:
        duplicate_count = int(df["hadm_id"].duplicated().sum())
        raise ValueError(f"Treatment feature hadm_id must be unique. Found {duplicate_count} duplicate rows.")

    df = df.set_index("hadm_id", drop=False)

    if validate:
        summary = validate_cdss_treatment_features(df)
        if not summary.is_valid:
            raise ValueError(f"Treatment validation failed: {summary}")

    if prepare_sets:
        prepare_treatment_similarity_columns(df)

    return df
