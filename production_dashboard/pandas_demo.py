import pandas as pd

# 1) CSV를 읽는다 — LOT 번호는 '글자'로 읽어서 앞자리 0이 사라지지 않게 한다
frame = pd.read_csv("production.csv", encoding="utf-8-sig", dtype={"lot_id": "string"})

# 2) 약속(데이터 계약)한 열이 모두 있는지 먼저 확인한다
required = {"lot_id", "work_date", "equipment_id", "inspected_qty", "good_qty"}
if missing := required - set(frame.columns):
    raise ValueError(f"필수 열 누락: {sorted(missing)}")

print("── 열마다 어떤 종류로 읽혔는지")
print(frame.dtypes)
print("── 앞 3행")
print(frame.head(3).to_string(index=False))

# 3) 장비별로 묶어서 '검사수량'과 '양품수량'을 각각 더한다
#    (정상 production.csv 전용 시연입니다. 오류 파일은 validate.py 로 먼저 검사합니다)
summary = frame.groupby("equipment_id", as_index=False)[["inspected_qty", "good_qty"]].sum()

# 4) 수율은 '합계끼리' 나눈다 — LOT별 수율을 평균 내는 것이 아니다
summary["yield_pct"] = summary["good_qty"] / summary["inspected_qty"] * 100

print("── 장비별 집계 (소수점은 화면에 보이는 모양만 2자리)")
print(summary.to_string(index=False, float_format="%.2f"))

# 5) 검산 — 장비별 합계를 다시 더하면 원본 전체 합계와 같아야 한다
total = frame["inspected_qty"].sum()
assert summary["inspected_qty"].sum() == total
print("검산 통과: 장비별 검사수량 합계 =", total, "= 원본 전체 합계")
