# OnBase Percentage Prediction

XGBoost-based professional-baseball probability prediction system. Deploy
`streamlit_app.py` as the Streamlit entry point for the pre-pitch, final
plate-appearance on-base probability model.

## Models

- **Plate-appearance model:** before each pitch, estimates the probability that
  the plate appearance ultimately ends with the batter reaching base.

The Streamlit app exposes both leagues in the left navigation:

- MLB live-game inference from the MLB Stats API.
- CPBL manual-state inference using compact 2018–2024 player profiles.

Both interfaces provide local TreeSHAP explanations and count-scenario tables.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Raw Statcast data, training reports, training-only outputs, and credentials are
intentionally excluded from the Streamlit runtime.

## Research materials

- [`docs/`](docs/) contains the latest full report, final one-page project
  resume, calibration curve, and feature-importance figure.
- [`results/`](results/) contains model comparison, temporal test metrics,
  hyperparameter search, ablation, feature-selection, and calibration artifacts.
- [`training/`](training/) contains the main training pipeline, deployment
  profile builder, and MLB player-name cache.

## CPBL adaptation

[`docs/CPBL_DATA.md`](docs/CPBL_DATA.md) documents the available CPBL sources,
the field alignment, and the data gaps. After exporting CPBL public game-log
events, run `python training/cpbl.py` to train the compatible pre-pitch model.

The deployable CPBL model, 73-feature manifest, metrics, and compact player
profiles live in [`cpbl_deployment/`](cpbl_deployment/). Rebuild the player
profiles after refreshing the raw CPBL data with:

```bash
python training/prepare_cpbl_deployment_data.py
```

The CPBL page deliberately uses manual game-state inputs so deployment does not
depend on a live third-party feed. Historical player features are precomputed;
the 138 MB raw game-log CSV is not required by Streamlit.

The raw `mlb_statcast_data.parquet` file is intentionally excluded because it is
about 326 MB. To retrain on another device, download the private backup and put
it beside `training/mlb.py`:

```text
training/
├─ mlb.py
├─ prepare_deployment_data.py
└─ mlb_statcast_data.parquet  # local only; ignored by Git
```

Run the training pipeline from that directory. Generated models and deployment
profiles should be reviewed before replacing the tested files in the repository
root.

## 2023 out-of-time results

| Model and target | Brier | Log Loss | ROC AUC | ECE |
|---|---:|---:|---:|---:|
| Plate-appearance final on-base probability | 0.20840 | 0.60586 | 0.60949 | 0.00605 |
