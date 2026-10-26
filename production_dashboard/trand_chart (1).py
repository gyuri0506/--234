from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 화면 없이도(원격·터미널 전용 환경) 그림 파일을 저장할 수 있게 한다
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트를 못 정해 주면 그래프 글자가 전부 네모(□)로 깨진다 — 설치된 것 중 하나를 고른다
설치된_폰트 = {f.name for f in font_manager.fontManager.ttflist}
for 후보 in ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"):
    if 후보 in 설치된_폰트:
        plt.rcParams["font.family"] = 후보
        break
else:
    print("한글 폰트를 찾지 못했습니다 — 그래프 글자가 네모로 보일 수 있습니다. 나눔고딕 등을 설치하세요.")
plt.rcParams["axes.unicode_minus"] = False

# 1) 날짜별로 묶어서 수율과 분모(검사수량)를 같이 계산한다
base = Path(__file__).resolve().parent
source = base / "원본" / "production.csv"
if not source.is_file():
    source = base / "production.csv"
frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"lot_id": "string"})
frame["work_date"] = pd.to_datetime(frame["work_date"], errors="raise")
daily = frame.groupby("work_date", as_index=False)[["inspected_qty", "good_qty"]].sum()
daily["yield_pct"] = daily["good_qty"] / daily["inspected_qty"] * 100

# 2) 날짜를 실제 '날짜형'으로 바꾸고, 빠진 날짜가 있으면 그 자리를 비워 둔다
#    (0으로 채우지 않는다 — '측정 안 함'과 '전부 불량'은 다른 이야기다)
daily["work_date"] = pd.to_datetime(daily["work_date"])
daily = daily.set_index("work_date").sort_index()
full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
daily = daily.reindex(full_range)  # 데이터가 없던 날짜는 NaN(결측)으로 남는다

missing = [d.strftime("%Y-%m-%d") for d in daily.index[daily["yield_pct"].isna()]]
if missing:
    print("결측(측정 없음 또는 수율 계산 불가):", ", ".join(missing))
else:
    print("결측 없음 — 이 구간은 전부 측정됨")

# 검사수량이 존재하는 날짜 중 가장 작은 날을 찾는다
if daily["inspected_qty"].notna().any():
    작은날 = daily["inspected_qty"].idxmin()
    print(f"분모(검사수량)가 가장 작은 날: {작은날:%Y-%m-%d} ({daily.loc[작은날, 'inspected_qty']:,.0f}개)")
else:
    print("검사수량이 있는 날짜가 없습니다.")

# 3) 그래프는 위아래 두 칸으로 나눈다 — 수율과 검사수량은 단위가 다르다
fig, (ax_yield, ax_qty) = plt.subplots(
    2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[2, 1]
)

ax_yield.plot(daily.index, daily["yield_pct"], marker="o", color="#1E2761", linewidth=2)
for d in daily.index[daily["yield_pct"].isna()]:
    ax_yield.axvspan(
        d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
        color="#C0392B", alpha=0.15
    )
ax_yield.set_ylabel("수율 (%)")
ax_yield.set_ylim(0, 100)
ax_yield.set_title("일자별 수율 추이 (빨간 구간 = 결측, 0%가 아님)")
ax_yield.grid(axis="y", alpha=0.3)

ax_qty.bar(daily.index, daily["inspected_qty"], color="#5B8DB8", width=0.6)
ax_qty.set_ylabel("검사수량(분모)")
ax_qty.set_xlabel("날짜")
ax_qty.grid(axis="y", alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()

out = base / "결과" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
out.mkdir(parents=True, exist_ok=False)
png_path = out / "수율_추이.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print("저장:", png_path)
