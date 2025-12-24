#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
import re
from scipy import stats
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{10}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
}

def infer_schema(df):
    schema = {}
    for col in df.columns:
        non_null = df[col].dropna()
        dtype = str(non_null.dtype)
        schema[col] = {
            "inferred_type": dtype,
            "unique": non_null.nunique()
        }
    return schema

def null_analysis(df):
    return {
        col: {
            "null_count": int(df[col].isna().sum()),
            "null_percent": float(df[col].isna().mean() * 100)
        } for col in df.columns
    }

def duplicate_analysis(df):
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percent": float(df.duplicated().mean() * 100)
    }

def outlier_analysis(df, method="iqr"):
    results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "zscore":
            z = np.abs(stats.zscore(series))
            outliers = int((z > 3).sum())
        else:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())

        results[col] = outliers

    return results

def pii_scan(df):
    findings = {}
    for col in df.columns:
        sample = df[col].astype(str).head(1000)
        findings[col] = []
        for name, pattern in PII_PATTERNS.items():
            if sample.str.contains(pattern, regex=True).any():
                findings[col].append(name)
    return findings

def health_score(nulls, dups, outliers):
    score = 100
    score -= sum(v["null_percent"] for v in nulls.values()) / len(nulls)
    score -= dups["duplicate_percent"]
    score -= sum(outliers.values()) * 0.1
    return round(max(score, 0), 2)

def profile(args):
    df = pd.read_csv(args.file, delimiter=args.delimiter, nrows=args.sample)
    report = {}

    report["schema"] = infer_schema(df)
    report["nulls"] = null_analysis(df)
    report["duplicates"] = duplicate_analysis(df)
    report["outliers"] = outlier_analysis(df, args.outliers)

    if args.security:
        report["pii"] = pii_scan(df)

    if args.health_score:
        report["health_score"] = health_score(
            report["nulls"], report["duplicates"], report["outliers"]
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))

def drift(args):
    base = pd.read_csv(args.baseline)
    new = pd.read_csv(args.new)

    drift_report = {}
    common_cols = set(base.columns) & set(new.columns)

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(base[col]):
            stat, p = stats.ks_2samp(base[col].dropna(), new[col].dropna())
            drift_report[col] = {
                "ks_stat": stat,
                "p_value": p,
                "drift_detected": p < 0.05
            }

    print(json.dumps(drift_report, indent=2))

def main():
    parser = argparse.ArgumentParser("CSV Profiler")
    sub = parser.add_subparsers()

    p = sub.add_parser("profile")
    p.add_argument("file")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--sample", type=int)
    p.add_argument("--outliers", choices=["iqr", "zscore"], default="iqr")
    p.add_argument("--security", action="store_true")
    p.add_argument("--health-score", action="store_true")
    p.add_argument("-o", "--output")
    p.set_defaults(func=profile)

    d = sub.add_parser("drift")
    d.add_argument("baseline")
    d.add_argument("new")
    d.set_defaults(func=drift)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
