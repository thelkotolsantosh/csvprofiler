# CSV Profiler
A CLI tool to profile CSV datasets for data quality, drift, and security risks.

## Features
- Schema inference
- Null and duplicate detection
- Outlier analysis (IQR / Z-score)
- Dataset health score
- Data drift detection (KS-test)
- PII and security-sensitive data detection

## Installation
pip install pandas numpy scipy

## Usage
Profile a dataset:
csvprofiler profile data.csv

Save report:
csvprofiler profile data.csv -o report.json

Enable security scan:
csvprofiler profile data.csv --security

Detect data drift:
csvprofiler drift baseline.csv new.csv

## Why this matters
Most ML failures originate from bad data. This tool shifts detection left,
before models, dashboards, or production systems break.

## Roadmap
- HTML report
- CI/CD integration
- Streaming support
- Schema enforcement
