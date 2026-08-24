"""Single public entry point for both baseball probability models."""

import streamlit as st


st.set_page_config(
    page_title="職棒打席機率預測系統",
    page_icon="⚾",
    layout="wide",
)

st.title("基於 XGBoost 的職棒打席機率預測系統")
st.subheader("Professional Baseball In-At-Bat Probability Prediction")

st.markdown(
    """
本網站整合兩個預測時間點不同、彼此獨立的模型。請從左側選單進入功能頁面。

### 打席最終上壘率

在每一球投出前，利用投打者歷史能力、目前球數、壘況、比分、前一球與投球序列，
預測該打席最後上壘的機率。頁面可選擇 MLB 日期、比賽、投打組合與特定打席時間點。

### Contact-instant 安打機率

球被擊入場內的接觸瞬間，利用擊球初速、仰角、投球品質、守備佈陣與比賽情境，
預測這顆球成為安打的機率。此模型排除最終落點、實際飛行距離及事後擊球類型，
避免把比賽結果發生後的資訊回填至預測。
"""
)

left, right = st.columns(2)
with left:
    st.metric("主模型 2023 ROC AUC", "0.6095")
    st.caption("預測最終上壘率；Brier Score 0.2084，ECE 0.0061。")
with right:
    st.metric("Contact-instant 2023 ROC AUC", "0.8665")
    st.caption("預測場內擊球成為安打；Brier Score 0.1357，ECE 0.0091。")

st.info(
    "兩個分數不能直接比較，因為標籤與可取得資訊的時間點不同。"
    "所有預測皆為統計關聯與機率估計，不代表因果效果。"
)
