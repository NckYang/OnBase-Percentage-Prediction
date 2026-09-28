"""Manual-state CPBL pre-pitch final on-base probability predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from explainability import explain_prediction


ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT = ROOT / "cpbl_deployment"


@st.cache_resource
def load_model() -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(DEPLOYMENT / "cpbl_xgb_obp_model.json")
    return model


@st.cache_data
def load_assets():
    features = json.loads((DEPLOYMENT / "cpbl_feature_manifest.json").read_text(encoding="utf-8"))["final_features"]
    defaults = json.loads((DEPLOYMENT / "cpbl_defaults.json").read_text(encoding="utf-8"))
    batters = pd.read_csv(DEPLOYMENT / "cpbl_batter_profiles.csv", dtype={"player_id": str})
    pitchers = pd.read_csv(DEPLOYMENT / "cpbl_pitcher_profiles.csv", dtype={"player_id": str})
    return batters, pitchers, features, defaults


def choose_player(label: str, profiles: pd.DataFrame) -> pd.Series:
    options = profiles.apply(lambda row: f"{row['name']}（{row['player_id']}）", axis=1).tolist()
    selected = st.selectbox(label, options)
    return profiles.iloc[options.index(selected)]


def build_features(state: dict, batter: pd.Series, pitcher: pd.Series, features: list[str]) -> pd.DataFrame:
    values = {name: 0.0 for name in features}
    base_code = state["on_1b"] + 2 * state["on_2b"] + 4 * state["on_3b"]
    count_code = state["balls"] * 3 + state["strikes"]
    pitcher_left = float(pitcher.is_left)
    batter_left = float(batter.is_left)
    values.update({
        "balls": state["balls"], "strikes": state["strikes"], "outs_when_up": state["outs"],
        "inning": state["inning"], "score_diff": state["score_diff"],
        "pitches_in_pa_before": state["pitches_in_pa_before"],
        "inning_score_diff": state["inning"] * state["score_diff"],
        "batting_order": state["batting_order"], "is_home_team": state["is_home_team"],
        "runners_on_base": state["on_1b"] + state["on_2b"] + state["on_3b"],
        "runner_in_scoring_position": int(state["on_2b"] or state["on_3b"]),
        "bases_loaded": int(state["on_1b"] and state["on_2b"] and state["on_3b"]),
        "is_full_count": int(state["balls"] == 3 and state["strikes"] == 2),
        "previous_pitch_was_ball": int(state["previous_pitch"] == "壞球"),
        "previous_pitch_was_strike": int(state["previous_pitch"] in {"好球", "揮空", "界外"}),
        "previous_pitch_scored": int(state["previous_pitch"] == "得分事件"),
        "previous_pitch_was_foul": int(state["previous_pitch"] == "界外"),
        "previous_pitch_was_whiff": int(state["previous_pitch"] == "揮空"),
        "previous_pitch_in_play": int(state["previous_pitch"] == "形成場內球"),
        "has_previous_pitch": int(state["previous_pitch"] != "無前一球"),
        "pitcher_game_pitches_before": state["pitcher_game_pitches_before"],
        "pitcher_game_strike_rate_before": state["pitcher_game_strike_rate_before"],
        "pitcher_game_ball_rate_before": state["pitcher_game_ball_rate_before"],
        "pitcher_game_whiff_rate_before": state["pitcher_game_whiff_rate_before"],
        "pitcher_game_foul_rate_before": state["pitcher_game_foul_rate_before"],
        "is_starting_pitcher": state["is_starting_pitcher"],
        "is_relief_pitcher": 1 - state["is_starting_pitcher"],
        "batter_is_left": batter_left, "pitcher_is_left": pitcher_left,
        "platoon_advantage": int(batter_left != pitcher_left),
        "age_bat": batter.age, "age_pit": pitcher.age,
        "batter_hist_rate": batter.hist_rate, "pitcher_hist_rate": pitcher.hist_rate,
        "batter_hist_log_pa": np.log1p(batter.plate_appearances),
        "pitcher_hist_log_bf": np.log1p(pitcher.batters_faced),
        "batter_hist_hit_rate": batter.hist_hit_rate, "pitcher_hist_hit_rate": pitcher.hist_hit_rate,
        "batter_hist_walk_hbp_rate": batter.hist_walk_hbp_rate,
        "pitcher_hist_walk_hbp_rate": pitcher.hist_walk_hbp_rate,
        "batter_recent_50_obp": batter.recent_50_obp, "batter_recent_100_obp": batter.recent_100_obp,
        "pitcher_recent_50_obp": pitcher.recent_50_obp, "pitcher_recent_100_obp": pitcher.recent_100_obp,
        "batter_vs_hand_rate": batter.vs_left_rate if pitcher_left == 1 else batter.vs_right_rate,
        "pitcher_vs_side_rate": pitcher.vs_left_rate if batter_left == 1 else pitcher.vs_right_rate,
        "batter_vs_hand_log_samples": batter.vs_left_log_samples if pitcher_left == 1 else batter.vs_right_log_samples,
        "pitcher_vs_side_log_samples": pitcher.vs_left_log_samples if batter_left == 1 else pitcher.vs_right_log_samples,
        "batter_days_since_prev_game": state["batter_days_since_prev_game"],
        "pitcher_days_since_prev_game": state["pitcher_days_since_prev_game"],
        "batter_prior_pa_this_game": state["batter_prior_pa_this_game"],
        "pitcher_batters_faced_before": state["pitcher_batters_faced_before"],
        "batter_times_faced_pitcher_before": state["batter_times_faced_pitcher_before"],
    })
    for code in range(12):
        values[f"count_is_{code}"] = int(code == count_code)
    for code in range(8):
        values[f"base_is_{code}"] = int(code == base_code)
    return pd.DataFrame([[values[name] for name in features]], columns=features, dtype=float)


st.set_page_config(page_title="CPBL 逐球上壘率預測", page_icon="⚾", layout="wide")
st.title("⚾ CPBL 投球前最終上壘率預測")
st.caption("不使用 TrackMan；以公開逐球事件、球員歷史表現與目前賽況進行預測。")

try:
    model = load_model()
    batters, pitchers, features, defaults = load_assets()
except FileNotFoundError as exc:
    st.error(f"缺少 CPBL 部署檔案：{exc}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    batter = choose_player("選擇打者", batters)
with c2:
    pitcher = choose_player("選擇投手", pitchers)

st.subheader("目前賽況")
c1, c2, c3, c4 = st.columns(4)
with c1:
    inning = st.number_input("局數", 1, 15, 1)
    outs = st.selectbox("出局數", [0, 1, 2])
    batting_order = st.selectbox("打序", list(range(1, 10)))
with c2:
    balls = st.selectbox("壞球數", [0, 1, 2, 3])
    strikes = st.selectbox("好球數", [0, 1, 2])
    previous_pitch = st.selectbox("前一球結果", ["無前一球", "壞球", "好球", "揮空", "界外", "形成場內球", "得分事件"])
with c3:
    on_1b = st.checkbox("一壘有人")
    on_2b = st.checkbox("二壘有人")
    on_3b = st.checkbox("三壘有人")
    is_home_team = st.checkbox("打者為主隊", value=True)
with c4:
    batting_score = st.number_input("攻方分數", 0, 50, 0)
    fielding_score = st.number_input("守方分數", 0, 50, 0)
    is_starting_pitcher = st.checkbox("目前為先發投手", value=True)

with st.expander("本場累積與休息日（可依實際狀態調整）"):
    a, b, c = st.columns(3)
    with a:
        pitches_in_pa = st.number_input("該打席目前已投球數", 0, 30, int(balls + strikes))
        pitcher_game_pitches = st.number_input("投手本場已投球數", 0, 200, 0)
        pitcher_bf = st.number_input("投手本場已面對打者數", 0, 60, 0)
        times_faced = st.number_input("本場已對戰此打者次數", 0, 10, 0)
    with b:
        strike_rate = st.slider("投手本場好球比例", 0.0, 1.0, float(defaults["game_strike_rate"]), 0.01)
        ball_rate = st.slider("投手本場壞球比例", 0.0, 1.0, float(defaults["game_ball_rate"]), 0.01)
        whiff_rate = st.slider("投手本場揮空比例", 0.0, 1.0, float(defaults["game_whiff_rate"]), 0.01)
        foul_rate = st.slider("投手本場界外比例", 0.0, 1.0, float(defaults["game_foul_rate"]), 0.01)
    with c:
        batter_rest = st.number_input("打者距上次出賽天數", 0, 30, 1)
        pitcher_rest = st.number_input("投手距上次出賽天數", 0, 30, 4)
        batter_prior_pa = st.number_input("打者本場先前打席數", 0, 10, 0)

state = {
    "balls": balls, "strikes": strikes, "outs": outs, "inning": inning,
    "score_diff": batting_score - fielding_score, "pitches_in_pa_before": pitches_in_pa,
    "batting_order": batting_order, "is_home_team": int(is_home_team),
    "on_1b": int(on_1b), "on_2b": int(on_2b), "on_3b": int(on_3b),
    "previous_pitch": previous_pitch, "pitcher_game_pitches_before": pitcher_game_pitches,
    "pitcher_game_strike_rate_before": strike_rate, "pitcher_game_ball_rate_before": ball_rate,
    "pitcher_game_whiff_rate_before": whiff_rate, "pitcher_game_foul_rate_before": foul_rate,
    "is_starting_pitcher": int(is_starting_pitcher),
    "batter_days_since_prev_game": batter_rest, "pitcher_days_since_prev_game": pitcher_rest,
    "batter_prior_pa_this_game": batter_prior_pa, "pitcher_batters_faced_before": pitcher_bf,
    "batter_times_faced_pitcher_before": times_faced,
}
current_features = build_features(state, batter, pitcher, features)
probability = float(model.predict(xgb.DMatrix(current_features))[0])

st.metric("目前狀態下，本打席最終上壘機率", f"{probability:.1%}")
st.caption("上壘定義：安打、保送或觸身球。模型測試年度為 2024，並非保證單一打席結果。")

with st.expander("為什麼模型得到這個機率？"):
    shap_rows, baseline_probability, reconstructed_probability = explain_prediction(model, current_features)
    m1, m2 = st.columns(2)
    m1.metric("模型基準機率", f"{baseline_probability:.1%}")
    m2.metric("SHAP 重建機率", f"{reconstructed_probability:.1%}")
    st.bar_chart(shap_rows.sort_values("shap_log_odds").set_index("feature")[["shap_log_odds"]], horizontal=True)
    st.dataframe(shap_rows, hide_index=True, width="stretch")

grid_rows = []
for grid_balls in range(4):
    row = {"壞球數": grid_balls}
    for grid_strikes in range(3):
        scenario = state | {"balls": grid_balls, "strikes": grid_strikes}
        row[f"{grid_strikes} 好球"] = float(model.predict(xgb.DMatrix(build_features(scenario, batter, pitcher, features)))[0])
    grid_rows.append(row)
st.subheader("不同球數下的最終上壘機率")
st.dataframe(pd.DataFrame(grid_rows).set_index("壞球數").style.format("{:.1%}").background_gradient(cmap="RdYlGn"), width="stretch")

st.info("目前為穩定展示版：球員歷史資料取自 2018–2024，當場賽況由使用者輸入；尚未自動連接 CPBL 即時比賽。")
