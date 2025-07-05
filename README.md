Auto-Retraining ML Pipeline with DVC + GitHub Actions

Automatically retrains ML models when data or code changes

## Tools Used

- Python (scikit-learn, pandas)
- DVC (Data Version Control)
- GitHub Actions (CI/CD)
- Versioned metrics, models, and reproducibility

## How it works

- Changes to `data/dataset.csv` or code retrigger the pipeline
- `dvc.yaml` defines the stages (train + evaluate)
- Metrics are logged in `metrics.txt`

## Run locally

dvc repro
