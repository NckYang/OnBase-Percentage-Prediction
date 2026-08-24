"""Standalone Streamlit UI for the independent contact-instant model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from post_contact_model import RAW_COLUMNS, add_engineered_features, transform_features


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "post_contact_outputs"


@st.cache_resource
def load_artifacts():
    manifest = json.loads((ARTIFACTS / "post_contact_feature_manifest.json").read_text(encoding="utf-8"))
    calibrator = json.loads((ARTIFACTS / "post_contact_calibrator.json").read_text(encoding="utf-8"))
    model = xgb.Booster()
    model.load_model(ARTIFACTS / "post_contact_hit_model.json")
    return model, manifest, calibrator


def base_row(manifest: dict) -> dict:
    row = {col: np.nan for col in RAW_COLUMNS}
    row.update({k: v for k, v in manifest["numeric_medians"].items() if k in row})
    row.update({
        "game_date": pd.Timestamp("2023-08-01"), "description": "hit_into_play",
        "events": None, "on_1b": None, "on_2b": None, "on_3b": None,
    })
    for col, choices in manifest["categories"].items():
        row[col] = next((x for x in choices if x != "MISSING"), "MISSING")
    return row


def select_category(label: str, col: str, manifest: dict, preferred: str | None = None) -> str:
    choices = [x for x in manifest["categories"][col] if x != "MISSING"]
    index = choices.index(preferred) if preferred in choices else 0
    return st.selectbox(label, choices, index=index)


st.set_page_config(page_title="Contact-instant Hit Probability", page_icon="⚾", layout="wide")
st.title("接觸瞬間安打機率｜Contact-instant Hit Probability")
st.caption("此頁使用獨立模型，只使用擊球接觸瞬間已知資訊；不載入或修改原本的打席最終上壘率模型。")

try:
    model, manifest, calibrator = load_artifacts()
except FileNotFoundError:
    st.error("找不到 post_contact_outputs。請先執行 `python post_contact_model.py`。")
    st.stop()

row = base_row(manifest)
left, middle, right = st.columns(3)
with left:
    st.subheader("擊球品質")
    row["launch_speed"] = st.number_input("擊球初速 Exit velocity (mph)", 20.0, 125.0, 92.0, .5)
    row["launch_angle"] = st.number_input("擊球仰角 Launch angle (°)", -90.0, 90.0, 15.0, 1.0)
    st.caption("不使用最終落點、實際飛行距離或事後擊球類型。")
with middle:
    st.subheader("投球與對戰")
    row["stand"] = select_category("打者慣用側", "stand", manifest, "R")
    row["p_throws"] = select_category("投手慣用側", "p_throws", manifest, "R")
    row["pitch_type"] = select_category("球種", "pitch_type", manifest, "FF")
    row["release_speed"] = st.number_input("球速 (mph)", 40.0, 110.0, 93.0, .5)
    row["release_spin_rate"] = st.number_input("轉速 (rpm)", 500.0, 4000.0, 2250.0, 25.0)
    row["pfx_x"] = st.number_input("水平位移 pfx_x (ft)", -3.0, 3.0, 0.0, .05)
    row["pfx_z"] = st.number_input("垂直位移 pfx_z (ft)", -3.0, 3.0, 1.0, .05)
    row["plate_x"] = st.number_input("進壘水平位置 plate_x (ft)", -3.0, 3.0, 0.0, .05)
    row["plate_z"] = st.number_input("進壘高度 plate_z (ft)", 0.0, 6.0, 2.5, .05)
with right:
    st.subheader("守備與情境")
    row["home_team"] = select_category("球場／主隊", "home_team", manifest)
    row["if_fielding_alignment"] = select_category("內野佈陣", "if_fielding_alignment", manifest, "Standard")
    row["of_fielding_alignment"] = select_category("外野佈陣", "of_fielding_alignment", manifest, "Standard")
    row["inning_topbot"] = select_category("局數上下半", "inning_topbot", manifest, "Top")
    row["inning"] = st.number_input("局數", 1, 20, 5)
    row["outs_when_up"] = st.selectbox("出局數", [0, 1, 2])
    row["bat_score"] = st.number_input("攻方分數", 0, 30, 2)
    row["fld_score"] = st.number_input("守方分數", 0, 30, 2)
    bases = st.multiselect("壘上跑者", ["一壘", "二壘", "三壘"])
    row["on_1b"] = 1 if "一壘" in bases else None
    row["on_2b"] = 1 if "二壘" in bases else None
    row["on_3b"] = 1 if "三壘" in bases else None

# Remaining pitch measurements use training medians, making missing assumptions explicit.
frame = add_engineered_features(pd.DataFrame([row]))
features = transform_features(frame, manifest).reindex(columns=manifest["feature_names"], fill_value=0)
dmatrix = xgb.DMatrix(features, feature_names=manifest["feature_names"])
best_iteration = int(model.attr("best_iteration") or 0)
raw_probability = float(model.predict(dmatrix, iteration_range=(0, best_iteration + 1))[0])
if calibrator["method"] == "platt":
    score = calibrator["coef"] * raw_probability + calibrator["intercept"]
    probability = float(1 / (1 + np.exp(-score)))
else:
    probability = raw_probability

st.divider()
st.metric("預測成為安打的機率", f"{probability:.1%}")
st.progress(min(max(probability, 0.0), 1.0))
st.caption(f"部署校準方式：{calibrator['method']}（由 2022 下半年校準集決定；目前原始機率較佳）")

with st.expander("這次預測的 SHAP 解釋"):
    contributions = model.predict(dmatrix, pred_contribs=True, iteration_range=(0, best_iteration + 1))[0]
    local = pd.DataFrame({"特徵": manifest["feature_names"], "SHAP（log-odds）": contributions[:-1]})
    local["影響方向"] = np.where(local["SHAP（log-odds）"] >= 0, "提高安打機率", "降低安打機率")
    st.dataframe(local.reindex(local["SHAP（log-odds）"].abs().sort_values(ascending=False).index).head(12), hide_index=True)
    st.caption("SHAP 顯示各特徵把本次預測推高或壓低多少；它解釋模型關聯，不等同因果效果。")

st.info("方法限制：現有資料沒有擊球出射方位角、守備員逐幀座標、Sprint Speed、天氣與風向，因此目前無法描述球往哪個方向飛，也不能計算守備員到達落點的時間。")
