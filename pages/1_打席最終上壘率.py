from datetime import date
from pathlib import Path
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from explainability import explain_prediction


ROOT = Path(__file__).resolve().parent.parent
API_ROOT = "https://statsapi.mlb.com/api"
DESCRIPTION_CODES = {
    name: code for code, name in enumerate(
        ["automatic_ball", "automatic_strike", "ball", "blocked_ball",
         "bunt_foul_tip", "called_strike", "foul", "foul_bunt",
         "foul_pitchout", "foul_tip", "hit_by_pitch", "hit_into_play",
         "missed_bunt", "pitchout", "swinging_strike",
         "swinging_strike_blocked"], start=1
    )
}
WHIFFS = {"missed_bunt", "swinging_strike", "swinging_strike_blocked"}
FOULS = {"bunt_foul_tip", "foul", "foul_bunt", "foul_pitchout", "foul_tip"}
PITCH_FAMILY = {
    "FF": "fastball", "FA": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "ST": "breaking", "CU": "breaking", "KC": "breaking",
    "SV": "breaking", "CS": "breaking", "CH": "offspeed", "FS": "offspeed",
    "FO": "offspeed", "KN": "offspeed", "EP": "offspeed", "SC": "offspeed",
    "PO": "other",
}


def api_get(path, params=None):
    url = f"{API_ROOT}{path}"
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "MLB-OBP-research-app/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


@st.cache_data(ttl=30)
def get_schedule(game_date):
    payload = api_get("/v1/schedule", {"sportId": 1, "date": game_date})
    games = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away = game["teams"]["away"]["team"]["name"]
            home = game["teams"]["home"]["team"]["name"]
            status = game["status"]["detailedState"]
            games.append({
                "game_pk": game["gamePk"],
                "label": f"{away} @ {home} — {status}",
            })
    return games


@st.cache_data(ttl=8)
def get_live_feed(game_pk):
    return api_get(f"/v1.1/game/{game_pk}/feed/live")


@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(ROOT / "xgb_obp_model.json")
    return model


@st.cache_data
def load_profiles():
    batters = pd.read_csv(ROOT / "deployment_batter_profiles.csv").set_index("player_id")
    pitchers = pd.read_csv(ROOT / "deployment_pitcher_profiles.csv").set_index("player_id")
    features = json.loads((ROOT / "model_feature_manifest.json").read_text(encoding="utf-8"))["final_features"]
    defaults = json.loads((ROOT / "deployment_defaults.json").read_text(encoding="utf-8"))
    return batters, pitchers, features, defaults


def nested(data, *keys, default=None):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def pitch_events(play):
    return [event for event in play.get("playEvents", []) if event.get("isPitch")]


def plate_appearance_options(feed):
    options = []
    matchup_counts = {}
    for index, play in enumerate(feed["liveData"]["plays"].get("allPlays", [])):
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        batter = nested(matchup, "batter", "fullName", default="Unknown batter")
        pitcher = nested(matchup, "pitcher", "fullName", default="Unknown pitcher")
        batter_id = nested(matchup, "batter", "id")
        pitcher_id = nested(matchup, "pitcher", "id")
        if batter_id is None or pitcher_id is None:
            continue
        pair = (int(pitcher_id), int(batter_id))
        matchup_counts[pair] = matchup_counts.get(pair, 0) + 1
        result = nested(play, "result", "description", default="進行中")
        half = "上" if about.get("isTopInning") else "下"
        options.append({
            "index": index,
            "pitcher_id": pair[0],
            "pitcher_name": pitcher,
            "batter_id": pair[1],
            "batter_name": batter,
            "matchup_number": matchup_counts[pair],
            "label": f"{about.get('inning', '?')}局{half}｜{batter} vs {pitcher}｜{result}",
        })
    return options


def pitch_timepoint_options(play):
    events = pitch_events(play)
    is_complete = bool(play.get("about", {}).get("isComplete"))
    maximum_previous = max(0, len(events) - 1) if is_complete else len(events)
    options = []
    for previous_count in range(maximum_previous + 1):
        if previous_count == 0:
            balls, strikes = 0, 0
        else:
            count = events[previous_count - 1].get("count", {})
            balls = min(int(count.get("balls", 0)), 3)
            strikes = min(int(count.get("strikes", 0)), 2)
        options.append({
            "previous_count": previous_count,
            "label": f"第 {previous_count + 1} 球投出前｜{balls}-{strikes}",
        })
    return options


def derive_live_state(feed, play_index, previous_pitch_count):
    live = feed["liveData"]
    plays = live["plays"]
    all_plays = plays.get("allPlays", [])
    current = all_plays[play_index]
    matchup = current["matchup"]
    batter_id = int(matchup["batter"]["id"])
    pitcher_id = int(matchup["pitcher"]["id"])
    batter_name = matchup["batter"]["fullName"]
    pitcher_name = matchup["pitcher"]["fullName"]
    all_current_events = pitch_events(current)
    events = all_current_events[:previous_pitch_count]
    latest = events[-1] if events else None
    description = nested(latest or {}, "details", "code", default="none")
    latest_details = (latest or {}).get("details", {})
    result_code = ("X" if latest_details.get("isInPlay") else
                   "B" if latest_details.get("isBall") else
                   "S" if latest_details.get("isStrike") else None)
    result_label = {"B": "壞球", "S": "好球／界外", "X": "進場"}.get(result_code, "無前一球")

    all_prior_pitches = []
    pitcher_batters_faced = 0
    times_faced = 0
    first_pitcher_for_side = None
    current_top = bool(current["about"].get("isTopInning"))
    for prior_index, play in enumerate(all_plays):
        if prior_index >= play_index:
            break
        same_side = bool(play["about"].get("isTopInning")) == current_top
        play_pitcher = int(play["matchup"]["pitcher"]["id"])
        if same_side and first_pitcher_for_side is None:
            first_pitcher_for_side = play_pitcher
        if play_pitcher == pitcher_id:
            pitcher_batters_faced += 1
            all_prior_pitches.extend(pitch_events(play))
            if int(play["matchup"]["batter"]["id"]) == batter_id:
                times_faced += 1

    all_prior_pitches.extend(events)
    if first_pitcher_for_side is None:
        first_pitcher_for_side = pitcher_id
    speeds = [nested(event, "pitchData", "startSpeed") for event in all_prior_pitches]
    speeds = [float(value) for value in speeds if value is not None]
    strike_flags = [bool(nested(event, "details", "isStrike", default=False)) for event in all_prior_pitches]
    descriptions = [nested(event, "details", "code", default="") for event in all_prior_pitches]
    swings = [value in WHIFFS or value in FOULS or value == "hit_into_play" for value in descriptions]

    previous_play = all_plays[play_index - 1] if play_index > 0 else None
    same_inning_previous = (
        previous_play is not None
        and previous_play.get("about", {}).get("inning") == current.get("about", {}).get("inning")
        and bool(previous_play.get("about", {}).get("isTopInning")) == current_top
    )
    previous_result = previous_play.get("result", {}) if previous_play else {}
    away_score = int(previous_result.get("awayScore", 0) or 0)
    home_score = int(previous_result.get("homeScore", 0) or 0)
    bat_score, field_score = (away_score, home_score) if current_top else (home_score, away_score)
    prior_count = nested(events[-1], "count", default={}) if events else {}
    state_balls = int(prior_count.get("balls", 0))
    state_strikes = int(prior_count.get("strikes", 0))
    if events:
        outs = int(prior_count.get("outs", 0))
    elif same_inning_previous:
        outs = int(previous_play.get("count", {}).get("outs", 0))
    else:
        outs = 0
    previous_matchup = previous_play.get("matchup", {}) if same_inning_previous else {}
    on_1b = int(bool(previous_matchup.get("postOnFirst")))
    on_2b = int(bool(previous_matchup.get("postOnSecond")))
    on_3b = int(bool(previous_matchup.get("postOnThird")))
    return {
        "batter_id": batter_id, "pitcher_id": pitcher_id,
        "batter_name": batter_name, "pitcher_name": pitcher_name,
        "batter_side": nested(matchup, "batSide", "code", default="R"),
        "pitcher_hand": nested(matchup, "pitchHand", "code", default="R"),
        "inning": int(current["about"]["inning"]),
        "outs": min(outs, 2),
        "balls": min(state_balls, 3),
        "strikes": min(state_strikes, 2),
        "bat_score": int(bat_score), "field_score": int(field_score),
        "on_1b": on_1b, "on_2b": on_2b, "on_3b": on_3b,
        "home_team": not current_top,
        "starter": pitcher_id == first_pitcher_for_side,
        "batters_faced": pitcher_batters_faced,
        "times_faced": times_faced,
        "pitch_count": len(events) + 1,
        "game_pitch_count": len(all_prior_pitches),
        "pitcher_rest": 4, "batter_rest": 1,
        "previous_result": result_label,
        "previous_description": description,
        "previous_plate_z": float(nested(latest or {}, "pitchData", "coordinates", "pZ", default=2.5) or 2.5),
        "previous_spin": float(nested(latest or {}, "pitchData", "breaks", "spinRate", default=2200) or 2200),
        "speed_change": (speeds[-1] - speeds[-2]) if len(speeds) >= 2 else 0.0,
        "prior_balls": sum(bool(nested(event, "details", "isBall", default=False)) for event in events),
        "prior_strikes": sum(bool(nested(event, "details", "isStrike", default=False)) for event in events),
        "prior_whiffs": sum(value in WHIFFS for value in [nested(event, "details", "code", default="") for event in events]),
        "prior_fouls": sum(value in FOULS for value in [nested(event, "details", "code", default="") for event in events]),
        "average_speed": float(np.mean([nested(event, "pitchData", "startSpeed") for event in events if nested(event, "pitchData", "startSpeed") is not None])) if events else 92.0,
        "distinct_pitch_types": len({nested(event, "details", "type", "code") for event in events if nested(event, "details", "type", "code")}),
        "game_average_speed": float(np.mean(speeds)) if speeds else 92.0,
        "game_strike_rate": float(np.mean(strike_flags)) if strike_flags else 0.50,
        "game_whiff_rate": float(np.mean([value in WHIFFS for value in descriptions])) if descriptions else 0.10,
        "game_swing_rate": float(np.mean(swings)) if swings else 0.47,
        "selected_play_index": play_index,
        "previous_pitch_count": previous_pitch_count,
        "plate_appearance_complete": bool(current.get("about", {}).get("isComplete")),
        "plate_appearance_result": nested(current, "result", "description", default="尚未結束"),
    }


def profile_row(table, player_id, defaults, entity):
    if player_id in table.index:
        return table.loc[player_id]
    return pd.Series({
        "hist_rate": defaults["rate"], "hist_hit_rate": defaults["hit_rate"],
        "hist_walk_rate": defaults["walk_rate"], "hist_strikeout_rate": defaults["strikeout_rate"],
        "hist_whiff_rate": defaults["whiff_rate"], "hist_chase_rate": defaults["chase_rate"],
        "woba": defaults["woba"], "xwoba": defaults["woba"], "age": defaults["age"],
        "plate_appearances" if entity == "batter" else "batters_faced": 0,
    })


def value(row, name, default):
    result = row.get(name, default)
    return default if pd.isna(result) else float(result)


def build_features(state, batter, pitcher, features, defaults, balls, strikes):
    values = {name: 0.0 for name in features}
    base_code = state["on_1b"] + 2 * state["on_2b"] + 4 * state["on_3b"]
    matchup_code = 2 * int(state["batter_side"] == "L") + int(state["pitcher_hand"] == "L")
    score_diff = state["bat_score"] - state["field_score"]
    values.update({
        "balls": balls, "strikes": strikes, "outs_when_up": state["outs"],
        "inning": state["inning"], "score_diff": score_diff,
        "inning_score_diff": state["inning"] * score_diff,
        "runners_on_base": state["on_1b"] + state["on_2b"] + state["on_3b"],
        "runner_in_scoring_position": int(state["on_2b"] or state["on_3b"]),
        "is_full_count": int(balls == 3 and strikes == 2),
        "pitch_count": state["pitch_count"], "is_home_team": int(state["home_team"]),
        "platoon_advantage": int(state["batter_side"] != state["pitcher_hand"]),
        "pitcher_is_left": int(state["pitcher_hand"] == "L"),
        "pitcher_batters_faced_before": state["batters_faced"],
        "batter_times_faced_pitcher_before": state["times_faced"],
        "is_starting_pitcher": int(state["starter"]),
        "age_bat": value(batter, "age", defaults["age"]),
        "age_pit": value(pitcher, "age", defaults["age"]),
        "n_priorpa_thisgame_player_at_bat": state["times_faced"],
        "pitcher_days_since_prev_game": state["pitcher_rest"],
        "batter_days_since_prev_game": state["batter_rest"],
        "has_previous_pitch": int(state["previous_result"] != "無前一球"),
        "previous_description_code": DESCRIPTION_CODES.get(state["previous_description"], 0),
        "previous_result_code": {"無前一球": 0, "壞球": 1, "好球／界外": 2, "進場": 3}[state["previous_result"]],
        "previous_was_ball": int(state["previous_result"] == "壞球"),
        "previous_was_whiff": int(state["previous_description"] in WHIFFS),
        "previous_plate_z": state["previous_plate_z"],
        "previous_release_spin_rate": state["previous_spin"],
        "previous_speed_change": state["speed_change"],
        "prior_ball_pitches": state["prior_balls"], "prior_strike_pitches": state["prior_strikes"],
        "prior_whiffs": state["prior_whiffs"], "prior_fouls": state["prior_fouls"],
        "prior_average_release_speed": state["average_speed"],
        "prior_distinct_pitch_types": state["distinct_pitch_types"],
        "pitcher_game_pitches_before": state["game_pitch_count"],
        "pitcher_game_average_speed_before": state["game_average_speed"],
        "pitcher_game_strike_rate_before": state["game_strike_rate"],
        "pitcher_game_whiff_rate_before": state["game_whiff_rate"],
        "pitcher_game_swing_rate_before": state["game_swing_rate"],
    })
    batter_pa = value(batter, "plate_appearances", 0)
    pitcher_bf = value(pitcher, "batters_faced", 0)
    for entity, row, samples in [("batter", batter, batter_pa), ("pitcher", pitcher, pitcher_bf)]:
        values[f"{entity}_hist_rate"] = value(row, "hist_rate", defaults["rate"])
        values[f"{entity}_hist_log_pa" if entity == "batter" else f"{entity}_hist_log_bf"] = np.log1p(samples)
        for metric, default_key in [("hit_rate", "hit_rate"), ("walk_rate", "walk_rate"), ("strikeout_rate", "strikeout_rate"), ("whiff_rate", "whiff_rate"), ("chase_rate", "chase_rate")]:
            values[f"{entity}_hist_{metric}"] = value(row, f"hist_{metric}", defaults[default_key])
        values[f"{entity}_hist_woba"] = value(row, "woba", defaults["woba"])
        values[f"{entity}_hist_xwoba"] = value(row, "xwoba", defaults["woba"])
        for window in [50, 100]:
            values[f"{entity}_recent_{window}_obp"] = value(row, f"recent_{window}_obp", defaults["rate"])
            values[f"{entity}_recent_{window}_log_samples"] = np.log1p(min(samples, window))
    values["batter_vs_hand_rate"] = values["batter_hist_rate"]
    values["pitcher_vs_side_rate"] = values["pitcher_hist_rate"]
    values["batter_vs_hand_log_pa"] = np.log1p(batter_pa / 2)
    values["pitcher_vs_side_log_bf"] = np.log1p(pitcher_bf / 2)
    expected_whiff = 0
    for family in ["fastball", "breaking", "offspeed", "other"]:
        usage = value(pitcher, f"{family}_usage", 0.25)
        whiff = value(batter, f"{family}_whiff_rate", defaults["whiff_rate"])
        values[f"pitcher_{family}_usage"] = usage
        values[f"batter_{family}_whiff_rate"] = whiff
        expected_whiff += usage * whiff
    values["expected_matchup_whiff_rate"] = expected_whiff
    for name in features:
        if name.startswith("count_is_"):
            values[name] = int(int(name.rsplit("_", 1)[1]) == balls * 4 + strikes)
        elif name.startswith("base_is_"):
            values[name] = int(int(name.rsplit("_", 1)[1]) == base_code)
        elif name.startswith("matchup_is_"):
            values[name] = int(int(name.rsplit("_", 1)[1]) == matchup_code)
        elif name.startswith("previous_description_is_"):
            values[name] = int(state["previous_description"] == name.removeprefix("previous_description_is_"))
    return pd.DataFrame([[values[name] for name in features]], columns=features)


st.set_page_config(page_title="MLB Live OBP Predictor", page_icon="⚾", layout="wide")
st.title("⚾ MLB 即時逐球上壘率預測")
st.caption("選擇比賽後，自動讀取目前投打組合與逐球 Statcast 資訊。")

model = load_model()
batters, pitchers, features, defaults = load_profiles()
selected_date = st.date_input("比賽日期", value=date(2023, 8, 1))
if st.button("重新整理即時資料"):
    get_schedule.clear()
    get_live_feed.clear()

try:
    games = get_schedule(selected_date.isoformat())
except Exception as exc:
    st.error(f"無法連接 MLB Stats API：{exc}")
    st.stop()
if not games:
    st.warning("這個日期沒有 MLB 比賽。")
    st.stop()

labels = [game["label"] for game in games]
selected_label = st.selectbox("選擇比賽", labels)
game_pk = games[labels.index(selected_label)]["game_pk"]
try:
    feed = get_live_feed(game_pk)
    pa_options = plate_appearance_options(feed)
    if not pa_options:
        st.warning("這場比賽目前還沒有打席資料。")
        st.stop()

    pitcher_names = {}
    pitcher_ids = []
    for option in pa_options:
        pitcher_names[option["pitcher_id"]] = option["pitcher_name"]
        if option["pitcher_id"] not in pitcher_ids:
            pitcher_ids.append(option["pitcher_id"])
    selected_pitcher_name = st.selectbox(
        "選擇投手",
        [pitcher_names[player_id] for player_id in pitcher_ids],
        index=len(pitcher_ids) - 1,
    )
    selected_pitcher_id = next(
        player_id for player_id in pitcher_ids
        if pitcher_names[player_id] == selected_pitcher_name
    )

    pitcher_pas = [option for option in pa_options if option["pitcher_id"] == selected_pitcher_id]
    batter_names = {}
    batter_ids = []
    for option in pitcher_pas:
        batter_names[option["batter_id"]] = option["batter_name"]
        if option["batter_id"] not in batter_ids:
            batter_ids.append(option["batter_id"])
    selected_batter_name = st.selectbox(
        "選擇打者",
        [batter_names[player_id] for player_id in batter_ids],
        index=len(batter_ids) - 1,
    )
    selected_batter_id = next(
        player_id for player_id in batter_ids
        if batter_names[player_id] == selected_batter_name
    )

    matchup_pas = [
        option for option in pitcher_pas
        if option["batter_id"] == selected_batter_id
    ]
    selected_matchup_number = st.selectbox(
        "選擇兩人本場第幾個打席",
        [option["matchup_number"] for option in matchup_pas],
        index=len(matchup_pas) - 1,
        format_func=lambda number: f"第 {number} 個打席",
    )
    selected_pa = next(
        option for option in matchup_pas
        if option["matchup_number"] == selected_matchup_number
    )
    play_index = selected_pa["index"]
    st.caption(selected_pa["label"])

    selected_play = feed["liveData"]["plays"]["allPlays"][play_index]
    timepoint_options = pitch_timepoint_options(selected_play)
    timepoint_labels = [option["label"] for option in timepoint_options]
    selected_timepoint_label = st.selectbox(
        "選擇該打席的時間點",
        timepoint_labels,
        index=len(timepoint_labels) - 1,
        help="每個選項代表該球投出前；模型只使用這個時間點以前已經發生的資訊。",
    )
    previous_pitch_count = timepoint_options[timepoint_labels.index(selected_timepoint_label)]["previous_count"]
    state = derive_live_state(feed, play_index, previous_pitch_count)
except Exception as exc:
    st.error(f"無法解析這場比賽的逐球資料：{exc}")
    st.stop()

batter = profile_row(batters, state["batter_id"], defaults, "batter")
pitcher = profile_row(pitchers, state["pitcher_id"], defaults, "pitcher")
st.info(
    f"預測時間點：第 {state['previous_pitch_count'] + 1} 球投出前。"
    "逐球與比賽情境特徵只使用此時間點以前的資料。"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("打者", state["batter_name"])
c2.metric("投手", state["pitcher_name"])
c3.metric("局數／出局", f"{state['inning']} 局 / {state['outs']} 出局")
c4.metric("目前球數", f"{state['balls']}-{state['strikes']}")
st.write(
    f"比分 **{state['bat_score']}–{state['field_score']}** ｜ "
    f"壘況 **{state['on_1b']}{state['on_2b']}{state['on_3b']}** ｜ "
    f"打者 **{state['batter_side']}** / 投手 **{state['pitcher_hand']}** ｜ "
    f"前一球：**{state['previous_description']}**"
)

current_features = build_features(
    state, batter, pitcher, features, defaults, state["balls"], state["strikes"]
)
current_probability = float(model.predict(xgb.DMatrix(current_features))[0])
st.metric("目前狀態下最終 OBP 成功機率", f"{current_probability:.1%}")

with st.expander("SHAP：為什麼模型得到這個機率？"):
    shap_rows, baseline_probability, reconstructed_probability = explain_prediction(
        model, current_features
    )
    st.markdown(
        "**簡單來說，SHAP 會把上方的預測機率拆開，顯示每項特徵把機率往上推或往下拉。**  "
        "\n- **模型基準機率**：模型解釋這筆資料時的共同起點，不是球員的歷史上壘率。"
        "\n- **正貢獻**：該特徵讓模型預測的上壘機率提高；**負貢獻**則讓機率降低。"
        "\n- **SHAP 重建機率**：把全部 97 個特徵的貢獻加回基準後得到的驗證值，"
        "應與上方原始預測相同，並不是第二個模型的預測。"
        "\n\n圖表只列出影響最大的 12 項，但重建機率使用全部 97 項。"
    )
    s1, s2 = st.columns(2)
    s1.metric("模型基準機率", f"{baseline_probability:.1%}")
    s2.metric("SHAP 重建機率（驗證）", f"{reconstructed_probability:.1%}")
    st.caption(
        "SHAP 的影響值使用 log-odds 尺度。它解釋模型如何形成預測，"
        "不代表某項特徵在真實比賽中造成上壘或出局。"
    )
    st.bar_chart(
        shap_rows.sort_values("shap_log_odds").set_index("feature")[["shap_log_odds"]],
        horizontal=True,
    )
    st.dataframe(
        shap_rows.rename(columns={
            "feature": "特徵",
            "feature_value": "當下數值",
            "shap_log_odds": "影響值（SHAP, log-odds）",
            "direction": "方向",
        }),
        hide_index=True,
        width="stretch",
    )

rows = []
for balls in range(4):
    row = {"壞球數": balls}
    for strikes in range(3):
        feature_row = build_features(state, batter, pitcher, features, defaults, balls, strikes)
        row[f"{strikes} 好球"] = float(model.predict(xgb.DMatrix(feature_row))[0])
    rows.append(row)
grid = pd.DataFrame(rows).set_index("壞球數")
st.subheader("若球數改變，最終 OBP 成功機率")
st.dataframe(
    grid.style.format("{:.1%}").background_gradient(cmap="RdYlGn", vmin=0.15, vmax=0.65),
    width="stretch",
)

st.subheader("該打席最後結果")
if state["plate_appearance_complete"]:
    st.success(state["plate_appearance_result"])
else:
    st.info("此打席尚未結束，目前沒有最終結果。")

with st.expander("自動取得的逐球特徵"):
    st.json({key: value for key, value in state.items() if key not in {"batter_name", "pitcher_name"}})

st.caption("資料來源：MLB Stats API。比賽進行中可按『重新整理即時資料』取得最新一球。")
