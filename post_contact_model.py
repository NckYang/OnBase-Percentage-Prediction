"""Independent contact-instant hit-probability model.

This module deliberately does not import ``mlb.py`` and never reads or writes the
production pre-pitch model artifacts. It predicts whether a ball put in play is
scored as a hit using only information available at the instant of contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "mlb_statcast_data.parquet"
DEFAULT_OUTPUT = ROOT / "post_contact_outputs"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "fielders_choice", "fielders_choice_out", "field_error", "sac_fly",
    "sac_fly_double_play", "triple_play",
}

RAW_COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "events", "description",
    "launch_speed", "launch_angle", "estimated_ba_using_speedangle", "hc_x",
    "hc_y", "hit_distance_sc", "launch_speed_angle", "bb_type", "stand",
    "p_throws", "home_team", "inning", "inning_topbot", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
    "if_fielding_alignment", "of_fielding_alignment", "pitch_type",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x",
    "plate_z", "zone", "effective_speed", "release_extension", "spin_axis",
]

CATEGORICAL = [
    "stand", "p_throws", "home_team", "inning_topbot",
    "if_fielding_alignment", "of_fielding_alignment", "pitch_type",
]

BASE_NUMERIC = [
    "launch_speed", "launch_angle",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x",
    "plate_z", "zone", "effective_speed", "release_extension", "spin_axis",
    "inning", "outs_when_up", "bat_score", "fld_score",
]


def load_contact_data(path: Path) -> pd.DataFrame:
    """Load only required columns and retain unambiguous balls in play."""
    df = pd.read_parquet(path, columns=RAW_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[
        df["description"].eq("hit_into_play")
        & df["events"].isin(HIT_EVENTS | OUT_EVENTS)
        & df["game_date"].notna()
    ].copy()
    df["target_hit"] = df["events"].isin(HIT_EVENTS).astype("int8")
    df["year"] = df["game_date"].dt.year
    return df.sort_values(["game_date", "game_pk", "at_bat_number"])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create contact physics and game-context features without target leakage."""
    z = df.copy()
    ev = pd.to_numeric(z["launch_speed"], errors="coerce")
    la = pd.to_numeric(z["launch_angle"], errors="coerce")

    z["launch_speed_sq"] = ev.pow(2)
    z["launch_angle_sq"] = la.pow(2)
    z["ev_x_launch_angle"] = ev * la
    z["hard_hit"] = (ev >= 95).astype(float).where(ev.notna())
    z["sweet_spot"] = la.between(8, 32).astype(float).where(la.notna())
    z["barrel_like"] = ((ev >= 98) & la.between(26, 30)).astype(float).where(ev.notna() & la.notna())
    z["score_diff"] = pd.to_numeric(z["bat_score"], errors="coerce") - pd.to_numeric(z["fld_score"], errors="coerce")
    z["runners_on"] = z[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)
    z["risp"] = (z["on_2b"].notna() | z["on_3b"].notna()).astype(float)
    z["bases_loaded"] = z[["on_1b", "on_2b", "on_3b"]].notna().all(axis=1).astype(float)
    z["same_hand"] = z["stand"].eq(z["p_throws"]).astype(float)
    z["movement_magnitude"] = np.hypot(
        pd.to_numeric(z["pfx_x"], errors="coerce"),
        pd.to_numeric(z["pfx_z"], errors="coerce"),
    )
    z["plate_radius"] = np.hypot(
        pd.to_numeric(z["plate_x"], errors="coerce"),
        pd.to_numeric(z["plate_z"], errors="coerce") - 2.5,
    )
    return z


ENGINEERED_NUMERIC = [
    "launch_speed_sq", "launch_angle_sq", "ev_x_launch_angle", "hard_hit", "sweet_spot",
    "barrel_like", "score_diff", "runners_on", "risp", "bases_loaded",
    "same_hand", "movement_magnitude", "plate_radius",
]


def fit_feature_manifest(train: pd.DataFrame) -> dict:
    medians = {}
    for col in BASE_NUMERIC + ENGINEERED_NUMERIC:
        value = pd.to_numeric(train[col], errors="coerce").median()
        medians[col] = 0.0 if pd.isna(value) else float(value)
    categories = {}
    for col in CATEGORICAL:
        values = train[col].fillna("MISSING").astype(str)
        categories[col] = sorted(values.unique().tolist())
    return {"numeric_medians": medians, "categories": categories}


def transform_features(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    parts = {}
    for col, median in manifest["numeric_medians"].items():
        parts[col] = pd.to_numeric(df[col], errors="coerce").fillna(median).astype("float32")
        parts[f"{col}__missing"] = df[col].isna().astype("float32")
    for col, cats in manifest["categories"].items():
        values = df[col].fillna("MISSING").astype(str)
        known = set(cats)
        for cat in cats:
            parts[f"{col}__{cat}"] = values.eq(cat).astype("float32")
        parts[f"{col}__UNKNOWN"] = (~values.isin(known)).astype("float32")
    return pd.DataFrame(parts, index=df.index)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = len(y)
    return float(sum(
        (ids == i).sum() / total * abs(y[ids == i].mean() - p[ids == i].mean())
        for i in range(bins) if (ids == i).any()
    ))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return {
        "n": int(len(y)), "hit_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece_15": expected_calibration_error(y, p, 15),
    }


def candidate_params() -> list[dict]:
    common = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "seed": 42, "nthread": -1}
    grid = [
        (4, .05, .8, .8, 1, 0), (5, .04, .8, .9, 2, 0),
        (6, .03, .8, .8, 3, .1), (4, .03, .9, .9, 3, .1),
        (7, .025, .75, .8, 5, .2), (5, .025, .9, .75, 5, .2),
        (6, .02, .9, .9, 8, .3), (3, .06, .85, .9, 1, 0),
    ]
    return [common | {"max_depth": d, "eta": eta, "subsample": sub, "colsample_bytree": col, "min_child_weight": mcw, "gamma": gamma}
            for d, eta, sub, col, mcw, gamma in grid]


def train(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = add_engineered_features(load_contact_data(data_path))
    train_df = df[df["year"].eq(2021)]
    year22 = df[df["year"].eq(2022)]
    cutoff = year22["game_date"].quantile(.5)
    val_df = year22[year22["game_date"] <= cutoff]
    cal_df = year22[year22["game_date"] > cutoff]
    test_df = df[df["year"].eq(2023)]
    if min(map(len, [train_df, val_df, cal_df, test_df])) == 0:
        raise ValueError("Expected non-empty 2021/2022/2023 chronological splits.")

    manifest = fit_feature_manifest(train_df)
    x_train = transform_features(train_df, manifest)
    manifest["feature_names"] = x_train.columns.tolist()
    x_val = transform_features(val_df, manifest).reindex(columns=x_train.columns, fill_value=0)
    x_cal = transform_features(cal_df, manifest).reindex(columns=x_train.columns, fill_value=0)
    x_test = transform_features(test_df, manifest).reindex(columns=x_train.columns, fill_value=0)
    y_train, y_val = train_df.target_hit.values, val_df.target_hit.values
    y_cal, y_test = cal_df.target_hit.values, test_df.target_hit.values
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=x_train.columns.tolist())
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=x_train.columns.tolist())
    dcal = xgb.DMatrix(x_cal, label=y_cal, feature_names=x_train.columns.tolist())
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=x_train.columns.tolist())

    search_rows, best = [], None
    for i, params in enumerate(candidate_params(), 1):
        model = xgb.train(params, dtrain, num_boost_round=2500, evals=[(dval, "validation")], early_stopping_rounds=80, verbose_eval=False)
        pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        row = {"candidate": i, "best_iteration": int(model.best_iteration), **{k: params[k] for k in ["max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "gamma"]}, **metrics(y_val, pred)}
        search_rows.append(row)
        print(f"candidate {i}/{len(candidate_params())}: validation logloss={row['logloss']:.6f}, rounds={row['best_iteration'] + 1}")
        if best is None or row["logloss"] < best[0]:
            best = (row["logloss"], model, params, row)
    assert best is not None
    model, best_params, best_row = best[1], best[2], best[3]

    raw_cal = model.predict(dcal, iteration_range=(0, model.best_iteration + 1))
    raw_test = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(np.clip(raw_cal, 1e-7, 1 - 1e-7).reshape(-1, 1), y_cal)
    calibrated_cal = platt.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
    use_platt = metrics(y_cal, calibrated_cal)["logloss"] < metrics(y_cal, raw_cal)["logloss"]
    calibrated_test = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    deployed_test = calibrated_test if use_platt else raw_test

    constant_test = np.full(len(y_test), y_train.mean())
    report = {
        "problem": "P(hit | ball put in play, information available at contact instant)",
        "positive_events": sorted(HIT_EVENTS), "negative_events": sorted(OUT_EVENTS),
        "splits": {
            "train": "2021", "validation": f"2022-01-01 through {cutoff.date()}",
            "calibration": f"after {cutoff.date()} through 2022-12-31", "test": "2023",
            "reserved_external": "2024 (not evaluated)",
        },
        "best_params": best_params, "best_iteration": int(model.best_iteration),
        "calibration_selected_on_2022": "platt" if use_platt else "none",
        "metrics": {
            "validation_raw": metrics(y_val, model.predict(dval, iteration_range=(0, model.best_iteration + 1))),
            "calibration_raw": metrics(y_cal, raw_cal),
            "calibration_platt": metrics(y_cal, calibrated_cal),
            "test_constant": metrics(y_test, constant_test),
            "test_raw": metrics(y_test, raw_test),
            "test_platt": metrics(y_test, calibrated_test),
            "test_deployed": metrics(y_test, deployed_test),
        },
    }
    xba_mask = test_df["estimated_ba_using_speedangle"].notna().values
    if xba_mask.any():
        xba = test_df.loc[xba_mask, "estimated_ba_using_speedangle"].astype(float).values
        report["metrics"]["test_statcast_xba_common_subset"] = metrics(y_test[xba_mask], xba)
        report["metrics"]["test_model_common_subset"] = metrics(y_test[xba_mask], deployed_test[xba_mask])

    model.save_model(output_dir / "post_contact_hit_model.json")
    manifest.update({
        "model_kind": "independent_contact_instant_hit_probability",
        "data_path_at_training": str(data_path.resolve()),
        "excluded_inputs": [
            "events", "estimated_ba_using_speedangle", "bat_speed", "hc_x", "hc_y",
            "hit_distance_sc", "launch_speed_angle", "bb_type",
        ],
        "calibration": "platt" if use_platt else "none",
        "target_definition": "1 for single/double/triple/home_run; 0 for listed non-hit balls in play",
    })
    (output_dir / "post_contact_feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "post_contact_calibrator.json").write_text(json.dumps({
        "method": "platt" if use_platt else "none", "coef": float(platt.coef_[0, 0]),
        "intercept": float(platt.intercept_[0]), "input": "raw_xgboost_probability",
    }, indent=2), encoding="utf-8")
    (output_dir / "post_contact_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(search_rows).sort_values("logloss").to_csv(output_dir / "post_contact_hyperparameter_search.csv", index=False)

    gain = model.get_score(importance_type="gain")
    pd.DataFrame({"feature": list(gain), "gain": list(gain.values())}).sort_values("gain", ascending=False).to_csv(output_dir / "post_contact_feature_importance.csv", index=False)
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_test, n_bins=12, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_test, n_bins=12, strategy="quantile")
    pd.DataFrame({"raw_mean_pred": prob_pred_raw, "raw_observed": prob_true_raw, "calibrated_mean_pred": prob_pred_cal, "calibrated_observed": prob_true_cal}).to_csv(output_dir / "post_contact_calibration_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred_raw, prob_true_raw, "o-", label="Raw XGBoost")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", label="Platt calibrated")
    ax.set(xlabel="Mean predicted hit probability", ylabel="Observed hit rate", title="2023 Contact-instant Hit Probability Calibration")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "post_contact_calibration_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved independent artifacts to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
