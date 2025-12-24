#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
import re
from scipy import stats
from datetime import datetime
----------------------------
# Security / PII patterns
----------------------------
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{10}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
}

 ----------------------------
# Core profiling functions
 ----------------------------
def infer_schema(df):
    schema = {}
    for col in df.columns:
        non_null = df[col].dropna()
        schema[col] = {
            "inferred_type": str(non_null.dtype),
            "unique_values": int(non_null.nunique())
        }
    return schema


def null_analysis(df):
    return {
        col: {
            "null_count": int(df[col].isna().sum()),
            "null_percent": round(df[col].isna().mean() * 100, 2)
        } for col in df.columns
    }


def duplicate_analysis(df):
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": dup_count,
        "duplicate_percent": round((dup_count / len(df)) * 100, 2) if len(df) else 0
    }


def outlier_analysis(df, method="iqr"):
    results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            results[col] = 0
            continue

        if method == "zscore":
            z = np.abs(stats.zscore(series))
            results[col] = int((z > 3).sum())
        else:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            results[col] = int(
                ((series < (q1 - 1.5 * iqr)) |
                 (series > (q3 + 1.5 * iqr))).sum()
            )

    return results


def pii_scan(df):
    findings = {}
    for col in df.columns:
        sample = df[col].astype(str).head(1000)
        detected = []
        for name, pattern in PII_PATTERNS.items():
            if sample.str.contains(pattern, regex=True, na=False).any():
                detected.append(name)
        if detected:
            findings[col] = detected
    return findings


def calculate_health_score(nulls, dups, outliers):
    score = 100

    avg_null = sum(v["null_percent"] for v in nulls.values()) / len(nulls)
    score -= avg_null
    score -= dups["duplicate_percent"]
    score -= sum(outliers.values()) * 0.1

    return round(max(score, 0), 2)

 ----------------------------
# HTML Dashboard Generator
 ----------------------------
def generate_html(report, output_file):
    labels = list(report["nulls"].keys())
    null_values = [v["null_percent"] for v in report["nulls"].values()]

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CSV Profiler Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 40px;
}}
h1, h2 {{
    color: #333;
}}
.card {{
    background: #fff;
    padding: 20px;
    margin-bottom: 30px;
    border-radius: 8px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    padding: 8px;
    border: 1px solid #ddd;
}}
th {{
    background: #eee;
}}
.good {{ color: green; font-weight: bold; }}
.bad {{ color: red; font-weight: bold; }}
</style>
</head>
<body>

<h1>CSV Profiler Report</h1>
<p>Generated at: {datetime.utcnow().isoformat()} UTC</p>

<div class="card">
<h2>Dataset Health</h2>
<p><b>Health Score:</b> {report.get("health_score", "N/A")}</p>
<p><b>Duplicate Rows:</b> {report["duplicates"]["duplicate_rows"]}</p>
</div>

<div class="card">
<h2>Schema Overview</h2>
<table>
<tr>
<th>Column</th>
<th>Type</th>
<th>Unique</th>
<th>Null %</th>
</tr>
"""

    for col, meta in report["schema"].items():
        html += f"""
<tr>
<td>{col}</td>
<td>{meta["inferred_type"]}</td>
<td>{meta["unique_values"]}</td>
<td>{report["nulls"][col]["null_percent"]}</td>
</tr>
"""

    html += """
</table>
</div>

<div class="card">
<h2>Null Distribution</h2>
<canvas id="nullChart"></canvas>
</div>

<script>
new Chart(document.getElementById("nullChart"), {
    type: "bar",
    data: {
        labels: %s,
        datasets: [{
            label: "Null Percentage",
            data: %s,
            backgroundColor: "rgba(255, 99, 132, 0.6)"
        }]
    }
});
</script>
""" % (labels, null_values)

    if "pii" in report:
        html += """
<div class="card">
<h2>Security / PII Findings</h2>
<ul>
"""
        for col, pii in report["pii"].items():
            html += f"<li class='bad'>{col}: {', '.join(pii)}</li>"
        html += "</ul></div>"

    html += """
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

# ----------------------------
# CLI Actions
# ----------------------------
def profile(args):
    df = pd.read_csv(args.file, delimiter=args.delimiter, nrows=args.sample)

    report = {
        "schema": infer_schema(df),
        "nulls": null_analysis(df),
        "duplicates": duplicate_analysis(df),
        "outliers": outlier_analysis(df, args.outliers)
    }

    if args.security:
        report["pii"] = pii_scan(df)

    if args.health_score:
        report["health_score"] = calculate_health_score(
            report["nulls"],
            report["duplicates"],
            report["outliers"]
        )

    if args.output:
        if args.output.endswith(".html"):
            generate_html(report, args.output)
        else:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


def drift(args):
    base = pd.read_csv(args.baseline)
    new = pd.read_csv(args.new)

    drift_report = {}

    for col in set(base.columns) & set(new.columns):
        if pd.api.types.is_numeric_dtype(base[col]):
            stat, p = stats.ks_2samp(
                base[col].dropna(),
                new[col].dropna()
            )
            drift_report[col] = {
                "ks_stat": round(stat, 4),
                "p_value": round(p, 6),
                "drift_detected": p < 0.05
            }

    print(json.dumps(drift_report, indent=2))

# ----------------------------
# Main
# ----------------------------
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
