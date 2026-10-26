"""Streamlit과 독립적인 CSV 검증·집계 함수. 파일 쓰기나 실행 부작용 없음."""

import csv
import io
import math
from collections import Counter, defaultdict
from datetime import date

from daily_yield import aggregate, create_cards


PRODUCTION_COLUMNS = ("lot_id", "work_date", "equipment_id", "inspected_qty", "good_qty", "defect_type")


def read_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("헤더가 없거나 열 이름이 중복되었습니다.")
    rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("CSV의 열 개수가 헤더와 다른 행이 있습니다.")
    return columns, rows


def csv_bytes(rows, columns=None):
    columns = list(columns or (list(rows[0]) if rows else []))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def validate_production(columns, rows):
    missing = set(PRODUCTION_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"필수 열 누락: {', '.join(sorted(missing))}")
    counts = Counter(str(r.get("lot_id", "")).strip() for r in rows)
    valid, rejected = [], []
    for line, original in enumerate(rows, 2):
        row = dict(original)
        reasons = []
        lot = str(row["lot_id"]).strip()
        if not lot or counts[lot] > 1:
            reasons.append("LOT 누락 또는 중복 그룹")
        try:
            raw = row["work_date"]
            if date.fromisoformat(raw).isoformat() != raw:
                raise ValueError()
        except (TypeError, ValueError):
            reasons.append("날짜 오류 (YYYY-MM-DD 필요)")
        quantities = [str(row[k]) for k in ("inspected_qty", "good_qty")]
        if not all(s.isascii() and s.isdecimal() for s in quantities):
            reasons.append("수량은 0 이상의 정수여야 함")
        else:
            inspected, good = map(int, quantities)
            if inspected <= 0 or good > inspected:
                reasons.append("검사수량은 양수, 양품수량은 검사수량 이하여야 함")
            else:
                row.update(inspected_qty=inspected, good_qty=good, defect_qty=inspected-good)
        if not str(row["equipment_id"]).strip():
            reasons.append("설비 누락")
        if reasons:
            rejected.append({"csv_row": line, **original, "reason": "; ".join(reasons)})
        else:
            row["lot_id"] = lot
            row["equipment_id"] = str(row["equipment_id"]).strip()
            row["defect_type"] = str(row["defect_type"]).strip() or "미분류"
            valid.append(row)
    return valid, rejected


def merge_lines(files, deduplicate=False):
    expected, merged, seen = None, [], set()
    for name, data in sorted(files):
        columns, rows = read_csv(data)
        if "source_file" in columns:
            raise ValueError(f"이미 source_file 열이 있습니다: {name}")
        if expected is None:
            expected = columns
        elif expected != columns:
            raise ValueError(f"열 구성 또는 순서가 다릅니다: {name}")
        for row in rows:
            key = tuple(row[c] for c in columns)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            merged.append({**row, "source_file": name})
    return merged


def equipment_summary(rows):
    groups = defaultdict(lambda: {"lot_count": 0, "inspected_qty": 0, "good_qty": 0})
    for row in rows:
        group = groups[row["equipment_id"]]
        group["lot_count"] += 1
        for key in ("inspected_qty", "good_qty"):
            group[key] += row[key]
    return [{"equipment_id": name, **group,
             "defect_qty": group["inspected_qty"]-group["good_qty"],
             "yield_pct": group["good_qty"]/group["inspected_qty"]*100}
            for name, group in sorted(groups.items())]


def pareto_summary(rows):
    groups = defaultdict(int)
    for row in rows:
        groups[row["defect_type"]] += row["defect_qty"]
    total = sum(groups.values())
    cumulative, summary = 0, []
    for name, qty in sorted(groups.items(), key=lambda pair: (-pair[1], pair[0])):
        cumulative += qty
        summary.append({"defect_type": name, "defect_qty": qty,
                        "share_pct": qty/total*100 if total else None,
                        "cumulative_qty": cumulative,
                        "cumulative_pct": cumulative/total*100 if total else None})
    if sum(r["defect_qty"] for r in summary) != sum(r["defect_qty"] for r in rows):
        raise ValueError("파레토 집계의 수량이 일치하지 않습니다.")
    return summary


def daily_summary(rows):
    if not rows:
        return [], []
    daily = aggregate(rows)
    segment = 0
    for row in daily:
        if not row["observed"]:
            segment += 1
        row["segment"] = segment
    return daily, create_cards(daily)


def join_quality(production, columns, quality):
    if "lot_id" not in columns:
        raise ValueError("품질 CSV에 lot_id 열이 필요합니다.")
    index = {r["lot_id"]: r for r in production}
    if len(index) != len(production):
        raise ValueError("생산 LOT가 중복되어 연결할 수 없습니다.")
    result = []
    for row in quality:
        lot = str(row["lot_id"]).strip()
        match = index.get(lot)
        result.append({**row, "production_equipment_id": match["equipment_id"] if match else None,
                       "production_matched": match is not None})
    return result


def calculation_checks(rows, rejected_count=0):
    """현재 선택 범위에서 실행한 계산 검증만 반환한다. 사람 승인과는 별개다."""
    labels = ["입력 데이터와 수량 조건", "설비별 검사수량 합계 보존", "전체 수율을 합계 기준으로 계산",
              "일별 양품수량 합계 보존", "유형별 불량수량 합계 보존", "마지막 누적 비중 100%"]
    if not rows:
        return [{"검사": label, "결과": "미실행", "확인 내용": "선택한 유효 데이터가 없습니다."} for label in labels]
    inspected = sum(r["inspected_qty"] for r in rows)
    good = sum(r["good_qty"] for r in rows)
    defects = sum(r["defect_qty"] for r in rows)
    equipment = equipment_summary(rows)
    daily, cards = daily_summary(rows)
    pareto = pareto_summary(rows)
    observations = [r for r in daily if r["observed"]]
    checks = [
        (not rejected_count and all(r["inspected_qty"]>0 and 0<=r["good_qty"]<=r["inspected_qty"] for r in rows),
         f"선택 {len(rows)} LOT · 전체 원본 보류 {rejected_count}행"),
        (sum(r["inspected_qty"] for r in equipment)==inspected, f"원본·설비 합계 {inspected:,}개"),
        (math.isclose(cards[4]["값"],good/inspected*100), f"{good:,} / {inspected:,} × 100 = {good/inspected*100:.2f}%"),
        (sum(r["good_qty"] for r in observations)==good, f"원본·일별 합계 {good:,}개"),
        (sum(r["defect_qty"] for r in pareto)==defects, f"원본·유형별 합계 {defects:,}개"),
        (math.isclose(pareto[-1]["cumulative_pct"],100.0) if defects else None,
         "100.00%" if defects else "불량수량 0개: 비중을 정의하지 않습니다."),
    ]
    return [{"검사":label, "결과":"미해당" if passed is None else ("통과" if passed else "보류"), "확인 내용":detail}
            for label,(passed,detail) in zip(labels,checks)]
