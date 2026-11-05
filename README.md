# Multimodal Intraoperative Physiological Coupling

This repository contains the analysis code associated with the manuscript:

**A Multimodal Physiological Coupling Framework for Intraoperative Arterial Pressure, Photoplethysmography, and Cardiorespiratory Signals**

## Overview

This study investigates multimodal physiological coupling among intraoperative arterial pressure (ART), photoplethysmography (PLETH), heart rate (HR), mean arterial pressure (MAP), and end-tidal carbon dioxide (ETCO2) signals.

Twelve coupling features were derived across four physiological coupling domains:

- ART-PLETH
- HR-MAP
- MAP-ETCO2
- HR-ETCO2

The repository includes code for signal preprocessing, quality control, coupling feature extraction, conventional physiological feature extraction, multivariable statistical analysis, sensitivity analyses, and figure generation.

## Data Source

The study uses the publicly available VitalDB database.

Raw physiological data are not redistributed in this repository. Users should obtain the required intraoperative biosignal and clinical data directly from VitalDB.

VitalDB: https://vitaldb.net/

## Analysis Workflow

1. Signal preprocessing and quality-control screening
2. Multimodal physiological coupling feature extraction
3. Conventional physiological feature extraction
4. Multivariable adjusted statistical analysis
5. Benjamini-Hochberg false discovery rate correction
6. Benjamini-Yekutieli FDR sensitivity analysis
7. Unique-subject sensitivity analysis
8. Alternative 20-minute analysis-window sensitivity analysis
9. Quality-control selection-bias assessment
10. Figure generation

## Repository Structure

```text
vitaldb-multimodal-physiological-coupling/
├── preprocessing/
├── feature_extraction/
├── analysis/
├── figures/
├── data_raw/
├── data_processed/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

### preprocessing

Contains scripts for signal-quality screening and feature-quality auditing.

### feature_extraction

Contains scripts for multimodal coupling feature extraction and conventional physiological feature calculation.

### analysis

Contains scripts for the primary adjusted analysis and sensitivity analyses, including unique-subject, multiple-testing, alternative-window, and quality-control selection-bias analyses.

### figures

Contains scripts used to generate manuscript figures.

## Data Directories

The following directories are excluded from the public repository through `.gitignore`:

- `data_raw/`
- `data_processed/`
- `results/`

These directories are intended for locally stored source data, intermediate files, and generated analysis outputs.

## Software

The analyses were performed using Python.

Required Python packages are listed in `requirements.txt`.

## Reproducibility

The preprocessing parameters, signal-quality criteria, coupling feature definitions, analysis windows, and statistical procedures are described in detail in the associated manuscript.

The scripts in this repository are provided to facilitate reproducibility of the preprocessing, feature extraction, statistical analysis, sensitivity analyses, and visualization procedures.

## Citation

Citation information will be added following publication of the associated manuscript.