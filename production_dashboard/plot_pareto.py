"""결과/defect_pareto.csv를 읽어 독립 SVG 파레토 차트를 저장한다."""

import csv
from html import escape
import math
from pathlib import Path


def main():
    folder = Path(__file__).parent / "결과"
    with (folder / "defect_pareto.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [int(row["defect_qty"]) for row in rows]
    if not values or sum(values) <= 0 or any(v < 0 for v in values):
        raise ValueError("차트를 그릴 양의 불량수량 합계가 필요합니다.")
    if values != sorted(values, reverse=True):
        raise ValueError("불량수량 내림차순 표가 필요합니다.")
    total = sum(values)
    cumulative = 0
    percentages = []
    for row, value in zip(rows, values):
        cumulative += value
        percent = cumulative / total * 100
        if not math.isclose(percent, float(row["cumulative_pct"]), abs_tol=1e-8):
            raise ValueError("표의 누적 비중과 불량수량이 일치하지 않습니다.")
        percentages.append(percent)

    left, right, top, bottom = 110, 940, 180, 590
    height = bottom - top
    maximum = math.ceil(max(values) / 200) * 200
    step = (right - left) / len(rows)
    centers = [left + step * (i + 0.5) for i in range(len(rows))]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="800" viewBox="0 0 1100 800">',
        '<rect width="1100" height="800" fill="#f7f9fc"/>',
        '<style>text{font-family:"Malgun Gothic","Noto Sans KR",sans-serif;fill:#24344b} .tick{font-size:14px;fill:#66758a}</style>',
        '<text x="70" y="60" font-size="30" font-weight="700">불량 유형별 파레토 차트</text>',
        f'<text x="70" y="94" font-size="17">집계 단위: 불량수량(개) · 총 {total:,}개 · 불량수량 내림차순</text>',
        '<rect x="70" y="119" width="18" height="12" rx="2" fill="#3274be"/>',
        '<text x="98" y="131" font-size="14">불량수량 (왼쪽 축)</text>',
        '<path d="M 300 125 H 335" stroke="#db6536" stroke-width="3"/>',
        '<circle cx="318" cy="125" r="4" fill="#db6536"/>',
        '<text x="347" y="131" font-size="14">누적 비중 (오른쪽 축)</text>',
        '<text x="110" y="163" font-size="14">불량수량 (개)</text>',
        '<text x="940" y="163" font-size="14" text-anchor="end">누적 비중 (%)</text>',
    ]
    for i in range(6):
        y = bottom - height * i / 5
        parts += [
            f'<path d="M {left} {y} H {right}" stroke="#dce3ed"/>',
            f'<text x="95" y="{y + 5}" text-anchor="end" class="tick">{maximum * i / 5:g}</text>',
            f'<text x="958" y="{y + 5}" class="tick">{20 * i}%</text>',
        ]
    threshold_y = bottom - height * 0.8
    parts += [
        f'<path d="M {left} {threshold_y} H {right}" stroke="#a2acba" stroke-dasharray="6 5"/>',
        f'<text x="1025" y="{threshold_y + 5}" font-size="13" text-anchor="middle">80% 기준</text>',
    ]
    for row, qty, x in zip(rows, values, centers):
        y = bottom - qty / maximum * height
        bar_width = min(130, step * 0.55)
        parts += [
            f'<rect x="{x - bar_width / 2}" y="{y}" width="{bar_width}" height="{bottom - y}" rx="4" fill="#3274be"/>',
            f'<text x="{x}" y="{y - 12}" text-anchor="middle" font-size="19" font-weight="700">{qty:,}개</text>',
            f'<text x="{x}" y="622" text-anchor="middle" font-size="17">{escape(row["defect_type"])}</text>',
            f'<text x="{x}" y="647" text-anchor="middle" class="tick">비중 {qty / total * 100:.2f}%</text>',
        ]
    points = [(x, bottom - pct / 100 * height) for x, pct in zip(centers, percentages)]
    coords = " ".join(f"{x},{y}" for x, y in points)
    parts.append(f'<polyline points="{coords}" fill="none" stroke="#db6536" stroke-width="3.5"/>')
    for (x, y), pct in zip(points, percentages):
        parts += [
            f'<circle cx="{x}" cy="{y}" r="6" fill="#db6536" stroke="white" stroke-width="2"/>',
            f'<rect x="{x + 10}" y="{y - 26}" width="88" height="26" rx="5" fill="#fff4ec"/>',
            f'<text x="{x + 54}" y="{y - 8}" text-anchor="middle" font-size="16" font-weight="700" style="fill:#af4822">{pct:.2f}%</text>',
        ]
    parts += [
        '<path d="M 110 180 V 590 H 940 V 180" stroke="#99a8bb" fill="none"/>',
        f'<text x="70" y="700" font-size="17" font-weight="700">상위 2개 유형의 누적 비중은 {percentages[min(1, len(rows)-1)]:.2f}%입니다.</text>',
        '<text x="70" y="735" font-size="14">불량수량 = 검사수량 − 양품수량 · LOT별 단일 유형에 불량수량 전체 배정</text>',
        '<text x="70" y="762" font-size="14">합성 실습 데이터 · 이 순위는 우선 검토 근거이며, 불량률 비교나 원인 증명이 아닙니다.</text>',
        '</svg>',
    ]
    destination = folder / "defect_pareto.svg"
    destination.write_text("\n".join(parts), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
