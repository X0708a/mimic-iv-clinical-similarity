from __future__ import annotations

import argparse

try:
    from .clinical_similarity import (
        find_clinically_similar_patients,
        load_clinical_similarity_dataset,
        recommend_treatments,
    )
    from .treatment_loader import load_cdss_treatment_features, validate_cdss_treatment_features
    from .treatment_similarity import find_similar_treatment_patients
except ImportError:  # pragma: no cover
    from clinical_similarity import (
        find_clinically_similar_patients,
        load_clinical_similarity_dataset,
        recommend_treatments,
    )
    from treatment_loader import load_cdss_treatment_features, validate_cdss_treatment_features
    from treatment_similarity import find_similar_treatment_patients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treatment and clinical similarity utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Load treatment features and print a quick summary")
    summary.add_argument("--db-path", default=None, help="Path to mimic.db")

    similar = subparsers.add_parser("similar", help="Find treatment-similar patients")
    similar.add_argument("--db-path", default=None, help="Path to mimic.db")
    similar.add_argument("--hadm-id", type=int, default=None, help="Query hadm_id")
    similar.add_argument("--k", type=int, default=10, help="Top-k similar patients")
    similar.add_argument("--no-stratify", action="store_true", help="Do not stratify by treatment intensity")

    clinical = subparsers.add_parser("clinical", help="Find diagnosis+treatment similar patients")
    clinical.add_argument("--db-path", default=None, help="Path to mimic.db")
    clinical.add_argument("--hadm-id", type=int, default=None, help="Query hadm_id")
    clinical.add_argument("--k", type=int, default=10, help="Top-k clinically similar patients")
    clinical.add_argument("--no-version-filter", action="store_true", help="Allow cross-version diagnosis comparisons")
    clinical.add_argument("--no-stratify", action="store_true", help="Do not stratify by treatment intensity")

    recommend = subparsers.add_parser("recommend", help="Recommend drugs from similar patients")
    recommend.add_argument("--db-path", default=None, help="Path to mimic.db")
    recommend.add_argument("--hadm-id", type=int, default=None, help="Query hadm_id")
    recommend.add_argument("--k", type=int, default=20, help="Top-k similar patients")
    recommend.add_argument("--top-n", type=int, default=10, help="Top-N recommended drugs")

    return parser


def run_summary(args: argparse.Namespace) -> None:
    df = load_cdss_treatment_features(db_path=args.db_path, include_raw_columns=False, validate=True, prepare_sets=True)
    summary = validate_cdss_treatment_features(df)
    print(f"rows={summary.row_count}")
    print(f"list_columns_are_python_lists={summary.list_columns_are_python_lists}")
    print(f"critical_null_counts={summary.critical_null_counts}")
    print(f"invalid_count_relationships={summary.invalid_count_relationships}")
    print(f"invalid_ratio_rows={summary.invalid_ratio_rows}")
    print(f"intensity_distribution={df['treatment_intensity_label'].value_counts().to_dict()}")
    print(f"avg_rx_unique_drugs={df['rx_unique_drugs'].mean():.4f}")
    print(f"avg_proc_count={df['proc_count'].mean():.4f}")


def run_similar(args: argparse.Namespace) -> None:
    df = load_cdss_treatment_features(db_path=args.db_path, include_raw_columns=False, validate=True, prepare_sets=True)
    query_hadm_id = args.hadm_id if args.hadm_id is not None else int(df["hadm_id"].iloc[0])
    results = find_similar_treatment_patients(
        df=df,
        query_hadm_id=query_hadm_id,
        k=args.k,
        stratify=not args.no_stratify,
        exclude_self=True,
    )
    print(f"query_hadm_id={query_hadm_id}")
    print(
        results[
            ["hadm_id", "treatment_intensity_label", "composite_similarity", "drug_sim", "proc_sim"]
        ].to_string(index=False)
    )


def run_clinical(args: argparse.Namespace) -> None:
    df = load_clinical_similarity_dataset(db_path=args.db_path)
    query_hadm_id = args.hadm_id if args.hadm_id is not None else int(df["hadm_id"].iloc[0])
    results = find_clinically_similar_patients(
        df=df,
        query_hadm_id=query_hadm_id,
        k=args.k,
        same_version_only=not args.no_version_filter,
        stratify_treatment=not args.no_stratify,
        exclude_self=True,
    )
    print(f"query_hadm_id={query_hadm_id}")
    print(
        results[
            [
                "hadm_id",
                "primary_diagnosis_icd",
                "treatment_intensity_label",
                "diagnosis_similarity",
                "treatment_similarity",
                "combined_similarity",
            ]
        ].to_string(index=False)
    )


def run_recommend(args: argparse.Namespace) -> None:
    df = load_clinical_similarity_dataset(db_path=args.db_path)
    query_hadm_id = args.hadm_id if args.hadm_id is not None else int(df["hadm_id"].iloc[0])
    results = recommend_treatments(
        df=df,
        query_hadm_id=query_hadm_id,
        k=args.k,
        top_n=args.top_n,
        use_clinical_similarity=True,
    )
    print(f"query_hadm_id={query_hadm_id}")
    print(results.to_string(index=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "summary":
        run_summary(args)
        return
    if args.command == "similar":
        run_similar(args)
        return
    if args.command == "clinical":
        run_clinical(args)
        return
    if args.command == "recommend":
        run_recommend(args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
