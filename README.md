# OnBase Percentage Prediction

XGBoost-based professional-baseball probability prediction system. Deploy
`streamlit_app.py` as the Streamlit entry point. One website contains two pages:

- Plate-appearance final on-base probability
- Contact-instant hit probability

## Models

- **Plate-appearance model:** before each pitch, estimates the probability that
  the plate appearance ultimately ends with the batter reaching base.
- **Contact-instant model:** after contact, estimates hit probability using only
  information available at contact. Final landing coordinates, actual flight
  distance, and post-play batted-ball classifications are excluded.

The two scores are not directly comparable because their targets and information
cutoffs differ. Both interfaces provide local TreeSHAP explanations.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Raw Statcast data, training reports, training-only outputs, and credentials are
intentionally excluded from the Streamlit runtime.

## Research materials

- [`docs/`](docs/) contains the latest full report, final one-page project
  resume, calibration curve, feature-importance figure, and the independent
  contact-instant model notes.
- [`results/`](results/) contains model comparison, temporal test metrics,
  hyperparameter search, ablation, feature-selection, calibration, and
  contact-instant evaluation artifacts.
- [`training/`](training/) contains the main training pipeline, deployment
  profile builder, and MLB player-name cache.

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
| Contact-instant hit probability | 0.13565 | 0.41830 | 0.86654 | 0.00910 |

These scores must not be compared as if they were the same task. The targets
and information available at prediction time are different.
