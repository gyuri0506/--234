"""불량수량 기준 파레토: python pareto.py [production.csv]"""

import math
import csv
import sys
from collections import defaultdict
from pathlib import Path

from validate import validate_rows


def summarize(rows):
    required = {"lot_id", "work_date", "equipment_id", "inspected_qty", "good_qty", "defect_type"}
    if not rows:
        raise ValueError("집계할 데이터가 없습니다.")
    if missing := required - set(rows[0]):
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    valid, rejected = validate_rows(rows)
    if rejected:
        raise ValueError(f"입력 오류: {rejected}")
    if not valid:
        raise ValueError("집계할 데이터가 없습니다.")
    # LOT 건수를 세지 않고, 각 LOT의 불량수량을 더한다.
    totals = defaultdict(int)
    for row in valid:
        row["defect_qty"] = row["inspected_qty"] - row["good_qty"]
        row["defect_type"] = (row["defect_type"] or "").strip() or "미분류"
        totals[row["defect_type"]] += row["defect_qty"]
    total = sum(row["defect_qty"] for row in valid)
    if total <= 0:
        raise ValueError("총 불량수량이 0이므로 비중과 누적 비중을 정의할 수 없습니다.")
    if sum(totals.values()) != total:
        raise ValueError("원본과 유형별 불량수량 합계가 다릅니다.")
    summary = []
    cumulative = 0
    for defect_type, qty in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        cumulative += qty
        summary.append(dict(defect_type=defect_type, defect_qty=qty,
                            share_pct=qty / total * 100, cumulative_qty=cumulative,
                            cumulative_pct=cumulative / total * 100))
    if not math.isclose(summary[-1]["cumulative_pct"], 100.0, abs_tol=1e-9):
        raise ValueError("마지막 누적 비중이 100%가 아닙니다.")
    return valid, summary


def check_hand_example():
    # 실행 전에 손으로 계산한 첫 5행: 27 + 108 + 70 + 45 + 106 = 356개.
    columns = {
        "lot_id": ["A", "B", "C", "D", "E"],
        "work_date": ["2026-10-01"] * 5,
        "equipment_id": ["EQ-SIM-01"] * 5,
        "inspected_qty": [1700, 850, 1700, 1000, 1000],
        "good_qty": [1673, 742, 1630, 955, 894],
        "defect_type": ["SIM-OTHER", "SIM-CHECK", "SIM-CHECK", "SIM-OTHER", "SIM-SURFACE"],
    }
    sample = [dict(zip(columns, values)) for values in zip(*columns.values())]
    frame, summary = summarize(sample)
    if [row["defect_qty"] for row in frame] != [27, 108, 70, 45, 106]:
        raise ValueError("손계산과 행별 불량수량이 다릅니다.")
    if [[row["defect_type"], row["defect_qty"]] for row in summary] != [
        ["SIM-CHECK", 178], ["SIM-SURFACE", 106], ["SIM-OTHER", 72]
    ] or [row["cumulative_qty"] for row in summary] != [178, 284, 356]:
        raise ValueError("손계산과 유형별 합계 또는 누적 합계가 다릅니다.")


def main():
    check_hand_example()
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("production.csv")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        frame = list(csv.DictReader(stream))
    frame, summary = summarize(frame)
    output = Path(__file__).parent / "결과"
    output.mkdir(exist_ok=True)
    for filename, rows in [("production_with_defect_qty.csv", frame), ("defect_pareto.csv", summary)]:
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    table = "defect_type    defect_qty  share_pct  cumulative_qty  cumulative_pct\n"
    table += "\n".join(
        f"{r['defect_type']:13} {r['defect_qty']:10} {r['share_pct']:10.2f} "
        f"{r['cumulative_qty']:15} {r['cumulative_pct']:15.2f}" for r in summary
    )
    report = (
        "집계 단위: 불량수량(개). defect_qty = inspected_qty - good_qty\n"
        "손계산 검산: 첫 5행 불량수량 27, 108, 70, 45, 106개.\n"
        "CHECK 178개, SURFACE 106개, OTHER 72개, 총 356개.\n"
        "손계산 표 누적 수량: 178, 284, 356개. 누적 비중: 50.00%, 79.78%, 100.00%.\n\n"
        + table
        + f"\n\n검산 통과: 원본과 유형별 불량수량 합계 {sum(r['defect_qty'] for r in frame)}개 일치. 마지막 누적 비중 100%.\n\n"
        "데이터 한계:\n"
        "- SYN/SIM 식별자를 가진 실습용 합성 데이터로, 실제 현장의 결론으로 일반화할 수 없습니다.\n"
        "- LOT당 유형이 하나뿐이므로, 해당 LOT의 불량수량 전부를 기록된 유형에 배정했습니다.\n"
        "  실제로 한 LOT에 여러 유형이 있다면 유형별 실제 불량수량 자료가 필요합니다.\n"
        "- 빈 유형은 미분류로 포함합니다. 입력 수량·날짜·LOT 오류는 집계 전에 중단합니다.\n"
        "- 수량 순위는 생산량, 검사량, 제품 구성, 설비 및 기간 차이를 보정하지 않습니다.\n"
        "  불량률 순위나 원인 증명이 아니라 우선 검토 근거입니다.\n"
        "- 마지막 누적 비중 100%만으로 원본의 누락이나 분류 정확성을 증명할 수 없습니다.\n"
        "  집계 전후 총량 일치를 함께 확인해도 원본에 없는 불량은 검증할 수 없습니다.\n"
        "- 계산값은 반올림하지 않고, 보고서 표시만 소수 둘째 자리로 제한했습니다.\n"
    )
    (output / "defect_pareto_notes.txt").write_text(report, encoding="utf-8-sig")
    print(report)


if __name__ == "__main__":
    main()
