from collections import Counter
from datetime import date
from pathlib import Path
import csv

def validate_rows(rows):
    counts = Counter(
        str(r.get("lot_id", "")).strip() for r in rows
    )

    valid, rejected = [], []

    for line, row in enumerate(rows, 2):
        reasons = []
        lot = str(row.get("lot_id", "")).strip()

        if not lot or counts[lot] > 1:
            reasons.append("LOT 누락 또는 중복 그룹")

        try:
            raw_date = row["work_date"]

            if date.fromisoformat(raw_date).isoformat() != raw_date:
                raise ValueError()

            quantities = [
                str(row[k]) for k in ("inspected_qty", "good_qty")
            ]

            if not all(
                s.isascii() and s.isdecimal()
                for s in quantities
            ):
                raise ValueError()

            checked, good = map(int, quantities)

            if checked <= 0 or good > checked:
                raise ValueError()

        except (KeyError, TypeError, ValueError):
            reasons.append("날짜 또는 수량 오류")

        equipment_id = str(
            row.get("equipment_id", "")
        ).strip()

        if not equipment_id:
            reasons.append("장비 누락")

        if reasons:
            rejected.append({
                "csv_row": line,
                "lot_id": lot,
                "reason": "; ".join(reasons)
            })
        else:
            valid.append({
                **row,
                "equipment_id": equipment_id,
                "inspected_qty": checked,
                "good_qty": good
            })

    return valid, rejected


# ==========================
# CSV 파일 자동 검색
# ==========================

base = Path(__file__).resolve().parent

candidates = [
    base / "원본" / "production_error_cases.csv",
    base / "production_error_cases.csv",
]

path = next((p for p in candidates if p.is_file()), None)

if path is None:
    print("오류: production_error_cases.csv 파일을 찾지 못했습니다.")
    print("파일을 아래 위치 중 하나에 넣어주세요.")

    for p in candidates:
        print("-", p)

else:
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    good, bad = validate_rows(rows)

    print("파일:", path)
    print("유효:", len(good), "보류:", len(bad))

    for item in bad:
        print(
            f"CSV {item['csv_row']}번째 줄 · "
            f"{item['lot_id']} · "
            f"{item['reason']}"
        )