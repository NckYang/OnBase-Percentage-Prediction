"""Streamlit entry point for the pre-pitch on-base probability model."""

import streamlit as st


st.set_page_config(
    page_title="投球前上壘機率預測",
    page_icon="⚾",
    layout="wide",
)

st.title("投球前上壘機率預測")
st.subheader("Pre-pitch On-base Probability Prediction")

st.markdown(
    """
本系統只使用投球前可取得的資訊，預測打者在本次打席最終成功上壘的機率。

請從左側選單選擇聯盟：

- **打席最終上壘率**：MLB 即時比賽與逐球資料。
- **CPBL 打席最終上壘率**：CPBL 球員與手動賽況展示版。

兩個頁面都會輸出預測機率、不同球數情境與特徵影響解釋。
"""
)

c1, c2 = st.columns(2)
c1.metric("MLB 2023 ROC AUC", "0.6095")
c2.metric("CPBL 2024 ROC AUC", "0.6144")
st.caption("時間外測試：MLB Brier Score 0.2084、ECE 0.0061；CPBL Brier Score 0.2137、ECE 0.0085。")

st.info("預測僅採用投球前資訊；不使用擊球後或比賽結束後才可取得的資料。")
