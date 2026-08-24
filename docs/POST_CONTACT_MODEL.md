# Independent contact-instant hit-probability model

This pipeline is physically and logically separate from the plate-appearance OBP
model. It does not import `mlb.py` and does not read or overwrite
`xgb_obp_model.json`, `model_feature_manifest.json`, or
`probability_calibrator.json`.

## Prediction target and timing

The target is `P(scored hit | ball put in play, information available at contact)`.
Singles, doubles, triples, and home runs are positive. Unambiguous non-hit balls
in play (including field outs, force outs, double plays, sacrifice flies,
fielder's choices, and reaching on error) are negative. Bunts and catcher's
interference are excluded from this first version.

`events` is used only to construct the label. Finalized landing coordinates
(`hc_x`, `hc_y`), actual flight distance (`hit_distance_sc`), post-contact batted
ball classification (`bb_type`), and their derived features are excluded.
Statcast
`estimated_ba_using_speedangle` is excluded from the model and retained only as
an external xBA benchmark. `bat_speed` is excluded because it is entirely absent
in the 2021–2022 training and validation periods.

## Time split

- Train: 2021
- Hyperparameter validation and early stopping: first half of 2022
- Calibration decision: second half of 2022
- Out-of-time test: 2023
- Reserved and not evaluated: 2024

All numeric medians and categorical vocabularies are learned from the 2021
training set only. Platt scaling was tested on the calibration split but rejected
because it worsened calibration-set log loss and ECE. The deployed probability is
therefore the raw XGBoost probability.

## 2023 results

| Model | Brier | Log loss | ROC AUC | PR AUC | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Constant training hit rate | 0.22131 | 0.63461 | 0.50000 | 0.33056 | 0.00394 |
| Independent contact-instant XGBoost | 0.13565 | 0.41830 | 0.86654 | 0.78003 | 0.00910 |

On the same 2023 rows where Statcast xBA is available, this model achieved Brier
0.13173 and ROC AUC 0.87421, versus xBA's 0.13230 and 0.87334. The nearly equal
results are plausible because both systems use contact quality; this model adds
pitch, alignment, park, and game context but lacks batted-ball direction.

## Leakage and deployment limitation

The earlier trajectory-aware experiment achieved Brier 0.05589, but it used
finalized landing coordinates and actual distance. Those fields made the problem
substantially easier and were too late for the stated prediction time, so they
were removed rather than reporting the score as a contact-instant result.

This dataset does not contain contact-time spray direction, defender frame
coordinates, Sprint Speed, weather, or wind. Adding them requires a source with
measured timestamps and a feature-availability cutoff, followed by a fresh
out-of-time validation.

## Commands

```bash
python post_contact_model.py
streamlit run post_contact_app.py
```

All artifacts are under `post_contact_outputs/` and begin with `post_contact_`.
