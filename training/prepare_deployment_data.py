from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "mlb_statcast_data.parquet"
PRIOR_PA = 100
PRIOR_OBP = 0.320
PITCH_FAMILY = {
    "FF": "fastball", "FA": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "ST": "breaking", "CU": "breaking", "KC": "breaking",
    "SV": "breaking", "CS": "breaking", "CH": "offspeed", "FS": "offspeed",
    "FO": "offspeed", "KN": "offspeed", "EP": "offspeed", "SC": "offspeed",
    "PO": "other",
}
SUCCESS = {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"}
FAILURE = {
    "strikeout", "field_out", "force_out", "grounded_into_double_play",
    "double_play", "strikeout_double_play", "fielders_choice",
    "fielders_choice_out", "field_error", "sac_fly", "sac_fly_double_play",
    "triple_play",
}
SWINGS = {
    "bunt_foul_tip", "foul", "foul_bunt", "foul_tip", "hit_into_play",
    "missed_bunt", "swinging_strike", "swinging_strike_blocked",
}
WHIFFS = {"missed_bunt", "swinging_strike", "swinging_strike_blocked"}


def smooth(successes, samples, prior_rate, prior_samples=PRIOR_PA):
    return (successes + prior_rate * prior_samples) / (samples + prior_samples)


def main():
    columns = [
        "game_date", "game_pk", "at_bat_number", "pitch_number", "events",
        "batter", "pitcher", "player_name", "stand", "p_throws", "pitch_type",
        "description", "zone", "woba_value", "estimated_woba_using_speedangle",
        "age_bat", "age_pit",
    ]
    df = pq.read_table(SOURCE, columns=columns).to_pandas()
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    valid_events = SUCCESS | FAILURE
    pa = df.loc[df["events"].isin(valid_events)].copy()
    pa["obp"] = pa["events"].isin(SUCCESS).astype("int8")
    pa["hit"] = pa["events"].isin({"single", "double", "triple", "home_run"}).astype("int8")
    pa["walk"] = pa["events"].isin({"walk", "intent_walk"}).astype("int8")
    pa["strikeout"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype("int8")
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"]).fillna(0)
    pa["woba"] = pa["woba_value"].fillna(0)

    batter = pa.groupby("batter").agg(
        plate_appearances=("obp", "size"), on_base=("obp", "sum"),
        hits=("hit", "sum"), walks=("walk", "sum"), strikeouts=("strikeout", "sum"),
        woba=("woba", "mean"), xwoba=("xwoba", "mean"), age=("age_bat", "last"),
    ).reset_index().rename(columns={"batter": "player_id"})
    batter_stand = pa.dropna(subset=["stand"]).groupby("batter")["stand"].agg(
        lambda values: "S" if values.nunique() > 1 else values.iloc[-1]
    )
    batter["stand"] = batter.player_id.map(batter_stand).fillna("R")
    batter["hist_rate"] = smooth(batter.on_base, batter.plate_appearances, PRIOR_OBP)
    batter["hist_hit_rate"] = smooth(batter.hits, batter.plate_appearances, 0.220)
    batter["hist_walk_rate"] = smooth(batter.walks, batter.plate_appearances, 0.085)
    batter["hist_strikeout_rate"] = smooth(batter.strikeouts, batter.plate_appearances, 0.225)

    pitcher = pa.groupby("pitcher").agg(
        batters_faced=("obp", "size"), on_base=("obp", "sum"),
        hits=("hit", "sum"), walks=("walk", "sum"), strikeouts=("strikeout", "sum"),
        woba=("woba", "mean"), xwoba=("xwoba", "mean"), age=("age_pit", "last"),
    ).reset_index().rename(columns={"pitcher": "player_id"})
    pitcher_hand = (
        pa.dropna(subset=["p_throws"]).groupby("pitcher")["p_throws"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[-1])
    )
    pitcher["p_throws"] = pitcher.player_id.map(pitcher_hand).fillna("R")
    pitcher["hist_rate"] = smooth(pitcher.on_base, pitcher.batters_faced, PRIOR_OBP)
    pitcher["hist_hit_rate"] = smooth(pitcher.hits, pitcher.batters_faced, 0.220)
    pitcher["hist_walk_rate"] = smooth(pitcher.walks, pitcher.batters_faced, 0.085)
    pitcher["hist_strikeout_rate"] = smooth(pitcher.strikeouts, pitcher.batters_faced, 0.225)

    for window in [50, 100]:
        recent_batter = pa.groupby("batter", sort=False).tail(window).groupby("batter")["obp"].agg(["sum", "count"])
        recent_pitcher = pa.groupby("pitcher", sort=False).tail(window).groupby("pitcher")["obp"].agg(["sum", "count"])
        batter[f"recent_{window}_obp"] = batter.player_id.map(
            smooth(recent_batter["sum"], recent_batter["count"], PRIOR_OBP, 25)
        ).fillna(PRIOR_OBP)
        pitcher[f"recent_{window}_obp"] = pitcher.player_id.map(
            smooth(recent_pitcher["sum"], recent_pitcher["count"], PRIOR_OBP, 25)
        ).fillna(PRIOR_OBP)

    pitch = df.loc[df["pitch_type"].notna()].copy()
    pitch["family"] = pitch["pitch_type"].map(PITCH_FAMILY).fillna("other")
    pitch["swing"] = pitch["description"].isin(SWINGS).astype("int8")
    pitch["whiff"] = pitch["description"].isin(WHIFFS).astype("int8")
    pitch["chase"] = (pitch["swing"].eq(1) & pitch["zone"].notna() & ~pitch["zone"].between(1, 9)).astype("int8")

    batter_pitch = pitch.groupby("batter").agg(
        pitch_samples=("pitch_type", "size"), whiffs=("whiff", "sum"), chases=("chase", "sum")
    )
    pitcher_pitch = pitch.groupby("pitcher").agg(
        pitch_samples=("pitch_type", "size"), whiffs=("whiff", "sum"), chases=("chase", "sum")
    )
    batter["hist_whiff_rate"] = batter.player_id.map(smooth(batter_pitch.whiffs, batter_pitch.pitch_samples, 0.105)).fillna(0.105)
    batter["hist_chase_rate"] = batter.player_id.map(smooth(batter_pitch.chases, batter_pitch.pitch_samples, 0.280)).fillna(0.280)
    pitcher["hist_whiff_rate"] = pitcher.player_id.map(smooth(pitcher_pitch.whiffs, pitcher_pitch.pitch_samples, 0.105)).fillna(0.105)
    pitcher["hist_chase_rate"] = pitcher.player_id.map(smooth(pitcher_pitch.chases, pitcher_pitch.pitch_samples, 0.280)).fillna(0.280)

    pitcher_family = pd.crosstab(pitch["pitcher"], pitch["family"])
    batter_family_count = pd.crosstab(pitch["batter"], pitch["family"])
    batter_family_whiff = pd.crosstab(pitch.loc[pitch.whiff.eq(1), "batter"], pitch.loc[pitch.whiff.eq(1), "family"])
    for family in ["fastball", "breaking", "offspeed", "other"]:
        p_count = pitcher_family.get(family, pd.Series(0, index=pitcher_family.index))
        p_total = pitcher_family.sum(axis=1)
        pitcher[f"{family}_usage"] = pitcher.player_id.map((p_count + 12.5) / (p_total + 50)).fillna(0.25)
        b_count = batter_family_count.get(family, pd.Series(0, index=batter_family_count.index))
        b_whiff = batter_family_whiff.get(family, pd.Series(0, index=batter_family_whiff.index))
        b_whiff = b_whiff.reindex(b_count.index, fill_value=0)
        batter[f"{family}_whiff_rate"] = batter.player_id.map((b_whiff + 3.15) / (b_count + 30)).fillna(0.105)

    pitcher_names = df.dropna(subset=["player_name"]).drop_duplicates("pitcher", keep="last").set_index("pitcher")["player_name"]
    pitcher["player_name"] = pitcher.player_id.map(pitcher_names).fillna(pitcher.player_id.astype(str))
    name_cache_path = ROOT / "mlb_player_names.json"
    if name_cache_path.exists():
        name_cache = {
            int(key): value for key, value in json.loads(
                name_cache_path.read_text(encoding="utf-8")
            ).items()
        }
        batter["player_name"] = batter.player_id.astype(int).map(name_cache)
    else:
        batter["player_name"] = batter.player_id.astype(str)
    batter["player_name"] = batter["player_name"].fillna(batter.player_id.astype(str))

    batter.to_csv(ROOT / "deployment_batter_profiles.csv", index=False)
    pitcher.to_csv(ROOT / "deployment_pitcher_profiles.csv", index=False)
    defaults = {
        "rate": PRIOR_OBP, "hit_rate": 0.220, "walk_rate": 0.085,
        "strikeout_rate": 0.225, "whiff_rate": 0.105, "chase_rate": 0.280,
        "woba": 0.315, "age": 28.0, "release_speed": 92.0,
        "spin_rate": 2200.0,
    }
    (ROOT / "deployment_defaults.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    print(f"Created {len(batter)} batter and {len(pitcher)} pitcher profiles.")


if __name__ == "__main__":
    main()
