from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="타이타닉 분석", layout="wide")
st.title("🚢 타이타닉 생존 분석")

# 저장소에 올려둔 그래프를 보여줘요.
for filename in [
    "titanic_survival.png",
    "titanic_hypothesis_charts.png",
    "titanic_hypothesis_summary.png",
]:
    image_path = BASE_DIR / filename
    if image_path.is_file():
        st.image(str(image_path))

# 저장소에 올려둔 분석 보고서를 보여줘요.
report_path = BASE_DIR / "kk"
if report_path.is_file():
    st.markdown(report_path.read_text(encoding="utf-8-sig"))
