"""일자별 수율·검사수량과 계산된 요약 카드 6항목을 생성한다."""

import csv
import json
from datetime import date, timedelta
from html import escape
from pathlib import Path
import sys
from collections import Counter


def aggregate(rows):
    if not rows:
        raise ValueError("집계할 데이터가 없습니다.")
    valid = []
    lots = Counter(str(r.get("lot_id", "")).strip() for r in rows)
    for line, row in enumerate(rows, 2):
        try:
            lot = str(row.get("lot_id", "")).strip()
            raw_date = row["work_date"]
            quantities = [str(row[k]) for k in ("inspected_qty", "good_qty")]
            if (not lot or lots[lot] != 1 or not str(row.get("equipment_id", "")).strip()
                    or date.fromisoformat(raw_date).isoformat() != raw_date
                    or not all(s.isascii() and s.isdecimal() for s in quantities)):
                raise ValueError()
            inspected, good = map(int, quantities)
            if inspected <= 0 or good > inspected:
                raise ValueError()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"CSV {line}행: LOT·날짜·설비·수량을 확인하세요.") from error
        valid.append({**row, "inspected_qty": inspected, "good_qty": good})
    grouped = {}
    for row in valid:
        day = date.fromisoformat(row["work_date"])
        item = grouped.setdefault(day, dict(work_date=day.isoformat(), lot_count=0,
                                           inspected_qty=0, good_qty=0))
        item["lot_count"] += 1
        item["inspected_qty"] += row["inspected_qty"]
        item["good_qty"] += row["good_qty"]
    result = []
    day, last = min(grouped), max(grouped)
    while day <= last:
        item = grouped.get(day)
        if item is None:
            item = dict(work_date=day.isoformat(), lot_count=None, inspected_qty=None,
                        good_qty=None, defect_qty=None, yield_pct=None, observed=False)
        else:
            item["defect_qty"] = item["inspected_qty"] - item["good_qty"]
            item["yield_pct"] = item["good_qty"] / item["inspected_qty"] * 100
            item["observed"] = True
        result.append(item)
        day += timedelta(days=1)
    for key in ("inspected_qty", "good_qty"):
        if sum(r[key] for r in result if r["observed"]) != sum(r[key] for r in valid):
            raise ValueError(f"일별 합계 불일치: {key}")
    if sum(r["lot_count"] for r in result if r["observed"]) != len(valid):
        raise ValueError("LOT 합계 불일치")
    return result


def create_cards(daily):
    observed = [r for r in daily if r["observed"]]
    inspected = sum(r["inspected_qty"] for r in observed)
    good = sum(r["good_qty"] for r in observed)
    low = min(r["yield_pct"] for r in observed)
    high = max(r["yield_pct"] for r in observed)
    def extrema(value):
        return [{k: r[k] for k in ("work_date", "yield_pct", "inspected_qty", "lot_count")}
                for r in observed if r["yield_pct"] == value]
    return [
        {"항목": "기간·관측일", "시작일": daily[0]["work_date"], "종료일": daily[-1]["work_date"],
         "관측일수": len(observed), "결측일수": len(daily)-len(observed),
         "LOT수": sum(r["lot_count"] for r in observed)},
        {"항목": "검사수량", "값": inspected, "단위": "개"},
        {"항목": "양품수량", "값": good, "단위": "개"},
        {"항목": "불량수량", "값": inspected-good, "단위": "개"},
        {"항목": "전체 수율", "값": good/inspected*100, "단위": "%", "산식": "양품수량 합계 / 검사수량 합계 × 100"},
        {"항목": "일별 최저·최고 수율", "최저": extrema(low), "최고": extrema(high)},
    ]


def draw(daily, cards, destination):
    left, right, top, bottom = 100, 1050, 180, 570
    height = bottom-top
    step = (right-left)/len(daily)
    max_qty = max(r["inspected_qty"] or 0 for r in daily)
    ceiling = ((max_qty+999)//1000)*1000
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1120" viewBox="0 0 1200 1120">',
           '<rect width="1200" height="1120" fill="#f7f9fc"/>',
           '<style>text{font-family:"Malgun Gothic","Noto Sans KR",sans-serif;fill:#24344b} .muted{fill:#67758a;font-size:14px}</style>']
    def text(x,y,value,size=16,extra=""):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>')
    text(65,55,"일자별 수율과 검사수량",30,'font-weight="700"')
    text(65,90,f'{daily[0]["work_date"]} ~ {daily[-1]["work_date"]} · 날짜 오름차순 · 수율 = 일별 양품 합계 / 검사 합계 × 100')
    svg.append('<rect x="65" y="116" width="20" height="12" fill="#a9c9ea"/>')
    text(95,128,"검사수량 (왼쪽 축)",14)
    svg.append('<path d="M 300 122 H 335" stroke="#db6536" stroke-width="3"/>')
    text(345,128,"수율 (오른쪽 축)",14)
    text(left,163,"검사수량 (개)",14)
    text(right,163,"수율 (%)",14,'text-anchor="end"')
    for i in range(6):
        y = bottom-height*i/5
        svg.append(f'<path d="M {left} {y} H {right}" stroke="#dce3ed"/>')
        text(left-14,y+5,f'{ceiling*i/5:,.0f}',14,'text-anchor="end"')
        text(right+15,y+5,f'{i*20}%',14)
    segment = []
    def finish_segment():
        if segment:
            points = " ".join(f'{x},{y}' for x,y in segment)
            svg.append(f'<polyline points="{points}" stroke="#db6536" stroke-width="3" fill="none"/>')
            segment.clear()
    # 결측일은 막대를 그리지 않고 선을 끊는다. 양쪽 축은 0에서 시작한다.
    for i,row in enumerate(daily):
        x = left+step*(i+.5)
        text(x,598,row["work_date"][5:],13,'text-anchor="middle"')
        if not row["observed"]:
            finish_segment()
            text(x,bottom-15,"결측",13,'text-anchor="middle"')
            continue
        bar_h = row["inspected_qty"]/ceiling*height
        width = min(48,step*.55)
        svg.append(f'<rect x="{x-width/2}" y="{bottom-bar_h}" width="{width}" height="{bar_h}" fill="#a9c9ea" rx="3"><title>{row["work_date"]}: 검사 {row["inspected_qty"]:,}개, {row["lot_count"]} LOT</title></rect>')
        text(x,bottom-bar_h-8,f'{row["inspected_qty"]:,}',12,'text-anchor="middle"')
        y = bottom-row["yield_pct"]/100*height
        segment.append((x,y))
    finish_segment()
    for i,row in enumerate(daily):
        if row["observed"]:
            x = left+step*(i+.5)
            y = bottom-row["yield_pct"]/100*height
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#db6536"><title>{row["work_date"]}: 수율 {row["yield_pct"]:.2f}%, 검사 {row["inspected_qty"]:,}개, {row["lot_count"]} LOT</title></circle>')
            text(x,y-12,f'{row["yield_pct"]:.2f}%',12,'text-anchor="middle"')
    text(65,640,"결측 규칙: 기간 내 기록이 없는 날은 수량·수율을 비우고 수율 선을 끊습니다. 0%로 대체하지 않습니다.",14)
    text(65,675,"계산된 요약 카드 · 6항목",22,'font-weight="700"')
    for i,card in enumerate(cards):
        x = 65+(i%3)*360
        y = 700+(i//3)*140
        svg.append(f'<rect x="{x}" y="{y}" width="345" height="125" rx="10" fill="white" stroke="#dce3ed"/>')
        text(x+16,y+28,card["항목"],16,'font-weight="700"')
        if i == 0:
            lines = [f'{card["시작일"]} ~ {card["종료일"]}',f'관측 {card["관측일수"]}일 · 결측 {card["결측일수"]}일 · {card["LOT수"]} LOT']
        elif i < 5:
            value = card["값"]
            lines = [f'{value:,.2f}%' if i == 4 else f'{value:,}개']
            if i == 4:
                lines.append("일별 수율의 평균이 아닌 합계 비율")
        else:
            lines = []
            for label in ("최저","최고"):
                r = card[label][0]
                lines += [f'{label} {r["work_date"][5:]} {r["yield_pct"]:.2f}% · {r["inspected_qty"]:,}개 / {r["lot_count"]} LOT']
            if any(len(card[label])>1 for label in ("최저","최고")):
                lines.append("동률 전체 날짜는 JSON 참조")
        for j,line in enumerate(lines):
            text(x+16,y+58+23*j,line,14)
    text(65,1015,"데이터 한계: 합성 실습 자료이며, 생산·검사 구성 차이를 보정하지 않았습니다. 추세는 원인 증명이 아닙니다.",14)
    text(65,1044,"결측은 '기록 없음'을 뜻하며 휴무·미생산을 확정하지 않습니다. 기간 양끝 밖의 미제공 날짜는 평가하지 않습니다.",14)
    text(65,1073,"AI 전달 규칙: daily_summary_cards.json의 계산값만 사용하고, 원본·누락값·원인을 추정하거나 보충하지 않습니다.",14)
    svg.append('</svg>')
    destination.write_text("\n".join(svg),encoding="utf-8")


def main():
    root = Path(__file__).parent
    source = Path(sys.argv[1]) if len(sys.argv)>1 else root/"production.csv"
    with source.open(encoding="utf-8-sig",newline="") as stream:
        daily = aggregate(list(csv.DictReader(stream)))
    cards = create_cards(daily)
    output = root/"결과"
    output.mkdir(exist_ok=True)
    with (output/"daily_yield.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    payload = {"요약카드": cards,
               "계산규칙": "일별 합산 후 날짜형 오름차순. 수율=양품 합계/검사 합계×100. 표시만 소수 둘째 자리 반올림.",
               "결측규칙": "최초~최종 관측일 사이의 무기록 날짜는 null. 수량·수율을 0으로 채우지 않고 선을 끊음. 기간 밖은 판단 불가.",
               "데이터한계": "합성 실습 자료. 기록 없음의 사유는 미상. 제품·설비·검사 구성 미보정. 원인 증명이 아님.",
               "AI전달규칙": "이 요약만 전달. 계산값을 그대로 인용하고 원본 수치·결측·원인을 추정하거나 보충하지 말 것."}
    (output/"daily_summary_cards.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    draw(daily,cards,output/"daily_yield.svg")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
