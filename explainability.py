"""Local TreeSHAP utilities shared by the Streamlit applications."""

import numpy as np
import pandas as pd
import xgboost as xgb


def explain_prediction(model, feature_frame, top_n=12):
    """Return the largest local SHAP contributions in model log-odds space."""
    matrix = xgb.DMatrix(feature_frame)
    contributions = model.predict(matrix, pred_contribs=True)[0]
    shap_values, bias = contributions[:-1], float(contributions[-1])
    explanation = pd.DataFrame({
        "feature": feature_frame.columns,
        "feature_value": feature_frame.iloc[0].to_numpy(dtype=float),
        "shap_log_odds": shap_values,
    })
    explanation["abs_shap"] = explanation["shap_log_odds"].abs()
    explanation["direction"] = np.where(
        explanation["shap_log_odds"] >= 0,
        "提高上壘機率",
        "降低上壘機率",
    )
    explanation = explanation.nlargest(top_n, "abs_shap").drop(columns="abs_shap")
    baseline_probability = float(1.0 / (1.0 + np.exp(-bias)))
    reconstructed_probability = float(
        1.0 / (1.0 + np.exp(-(bias + shap_values.sum())))
    )
    return explanation, baseline_probability, reconstructed_probability

