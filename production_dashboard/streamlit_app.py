"""실행: python -m streamlit run streamlit_app.py"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_core import (read_csv, csv_bytes, validate_production, merge_lines,
                            equipment_summary, pareto_summary, daily_summary, join_quality,
                            calculation_checks)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="생산·품질 작업실", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp { background: #ffffff; color: #262b36; }
  [data-testid="stSidebar"] { background: #f0f2f6; border-right: 1px solid #dfe2e8; }
  [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
  [data-testid="stSidebar"] h2 { font-size: 1rem; font-weight: 700; }
  [data-testid="stSidebar"] label p { font-size: .82rem; }
  .block-container { max-width: 1400px; padding: 2.9rem 3.2rem 2rem; }
  h1 { font-size: 2rem !important; letter-spacing: -.06em; padding-bottom: .3rem !important; }
  h2 { font-size: 1.25rem !important; letter-spacing: -.035em; }
  h3 { font-size: 1.05rem !important; }
  [data-testid="stCaptionContainer"] { color: #858994; }
  [data-testid="stMetricLabel"] p { color: #606773; font-size: .78rem; }
  [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 450; letter-spacing: -.04em; }
  [data-testid="stMetric"] { padding: .25rem 0 .65rem; }
  [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid #e6e8ed; }
  [data-baseweb="tab"] { height: 2.6rem; padding: 0; background: transparent; white-space: nowrap; }
  [data-baseweb="tab"] p { font-size: .8rem; }
  [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
  [data-testid="stExpander"] { border-color: #e1e5eb; border-radius: 5px; }
  [data-testid="stAlert"] { border-radius: 4px; font-size: .86rem; }
  div[data-testid="stDataFrame"] { border: 1px solid #e6e8ed; border-radius: 4px; }
  @media (max-width: 800px) {
    .block-container { padding: 2rem 1rem; }
    h1 { font-size: 1.65rem !important; }
    [data-baseweb="tab-list"] { gap: .9rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
  }
</style>
""", unsafe_allow_html=True)
st.title("생산·품질 작업실")
st.caption("합성 자료로 살펴보는 CSV 읽기 · 검증 · 연결 · 집계 · 시각화")


def table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def download(rows, filename, key, columns=None):
    st.download_button("CSV 다운로드", csv_bytes(rows, columns), filename,
                       mime="text/csv", key=key)


def chart(rows, spec):
    st.vega_lite_chart(pd.DataFrame(rows), spec, use_container_width=True)


def cards_view(cards):
    if not cards:
        return
    for i, column in enumerate(st.columns(3)):
        with column:
            with st.container(border=True):
                if i == 0:
                    c = cards[0]
                    st.metric(c["항목"], f'{c["관측일수"]}일 관측')
                    st.caption(f'{c["시작일"]} ~ {c["종료일"]} · 결측 {c["결측일수"]}일 · {c["LOT수"]} LOT')
                else:
                    c = cards[i]
                    st.metric(c["항목"], f'{c["값"]:,}개')
    for i, column in enumerate(st.columns(3), 3):
        with column:
            with st.container(border=True):
                c = cards[i]
                if i == 3:
                    st.metric(c["항목"], f'{c["값"]:,}개')
                elif i == 4:
                    st.metric(c["항목"], f'{c["값"]:.2f}%')
                    st.caption("양품 합계 / 검사 합계 × 100 (LOT 수율의 평균 아님)")
                else:
                    st.write("**일별 최저·최고 수율**")
                    for label in ("최저", "최고"):
                        for r in c[label]:
                            st.write(f'{label} {r["work_date"]}: **{r["yield_pct"]:.2f}%**')
                            st.caption(f'검사 {r["inspected_qty"]:,}개 · {r["lot_count"]} LOT')


st.sidebar.header("분석 범위")
with st.sidebar.expander("생산 CSV 바꾸기", expanded=False):
    uploaded = st.file_uploader("production.csv", type="csv", key="production")
try:
    raw = uploaded.getvalue() if uploaded is not None else (ROOT/"production.csv").read_bytes()
    columns, original = read_csv(raw)
    valid, rejected = validate_production(columns, original)
except (OSError, UnicodeError, ValueError) as error:
    st.error(f"생산 데이터 읽기 실패: {error}")
    st.stop()
if rejected:
    st.warning(f"오류가 있는 {len(rejected)}행을 보류했습니다. 데이터 검증 탭에서 사유를 확인하세요. 집계는 유효 행만 사용합니다.")
filtered = valid
equipment, date_range = [], ()
if valid:
    available = sorted({r["equipment_id"] for r in valid})
    equipment = st.sidebar.multiselect("설비", available, default=available[:1])
    dates = pd.to_datetime([r["work_date"] for r in valid])
    date_range = st.sidebar.date_input("작업 기간", (dates.min().date(), dates.max().date()),
                                       min_value=dates.min().date(), max_value=dates.max().date())
    if len(date_range) == 2:
        filtered = [r for r in valid if r["equipment_id"] in equipment
                    and date_range[0].isoformat() <= r["work_date"] <= date_range[1].isoformat()]
    else:
        st.info("분석 기간의 종료일을 선택하세요.")
        filtered = []
st.sidebar.caption("설비와 날짜를 고르면 생산 집계와 품질 연결이 함께 바뀝니다. CSV 병합·데이터 검증·처리일지는 각 원본 기준입니다.")
st.sidebar.info("합성 자료 전용 · 검토·실습 기준")
st.sidebar.caption(f'원본 {len(original)}행 · 유효 {len(valid)}행 · 보류 {len(rejected)}행')
daily, cards = daily_summary(filtered)
inspected_total = sum(r["inspected_qty"] for r in filtered)
good_total = sum(r["good_qty"] for r in filtered)
defect_total = sum(r["defect_qty"] for r in filtered)
for column, label, value in zip(st.columns(4),
                               ["선택 LOT", "검사수량", "불량수량", "수율"],
                               [f"{len(filtered):,}건", f"{inspected_total:,}개", f"{defect_total:,}개",
                                f"{good_total/inspected_total*100:.2f}%" if inspected_total else "—"]):
    column.metric(label, value)
tabs = st.tabs(["파일·작업 안내", "CSV 합치기", "데이터 검증", "장비별 수율", "LOT 연결", "불량 파레토", "수율 추이", "처리일지", "검증·사람 승인"])

with tabs[0]:
    if not filtered:
        st.info("선택 조건에 해당하는 유효 생산 데이터가 없습니다.")
    st.subheader("폴더와 기존 작업")
    table([
        {"위치": "production.csv / quality.csv", "역할": "생산·품질 원본", "관련 작업": "first_read.py / join.py"},
        {"위치": "라인/", "역할": "라인별 CSV 원본", "관련 작업": "파일 분류/my_mere.py / my_merge.py"},
        {"위치": "validate.py", "역할": "날짜·수량·LOT 검증 실습", "관련 작업": "검증 탭에서 오류 행 보류"},
        {"위치": "pandas_demo.py", "역할": "설비별 합계와 가중 수율", "관련 작업": "장비별 수율 탭"},
        {"위치": "pareto.py / plot_pareto.py", "역할": "불량수량 파레토", "관련 작업": "불량 파레토 탭"},
        {"위치": "trand_chart (1).py / daily_yield.py", "역할": "일별 수율·분모·결측 처리", "관련 작업": "일별 수율 탭"},
        {"위치": "로그/ / 결과/", "역할": "처리일지·계산 결과", "관련 작업": "add_duration.py / 처리일지 탭"},
        {"위치": "실습 1/", "역할": "기존 실습 안내", "관련 작업": "분석 데이터로 사용하지 않음"},
    ])
    st.caption("앱은 원본과 기존 결과 파일을 덮어쓰지 않습니다. 다운로드는 현재 선택한 데이터에서 계산합니다.")

with tabs[1]:
    st.subheader("라인별 CSV 행 병합")
    st.caption("라인/의 CSV를 합칩니다. source_file에 출처를 남깁니다. 같은 LOT라도 행 내용이 다르면 중복 제거하지 않습니다.")
    line_uploads = st.file_uploader("여러 라인 CSV 업로드 (선택)",type="csv",accept_multiple_files=True,key="lines")
    dedup = st.checkbox("완전히 같은 행만 중복 제거",value=True)
    try:
        files = [(f.name,f.getvalue()) for f in line_uploads] if line_uploads else [(p.name,p.read_bytes()) for p in sorted((ROOT/"라인").glob("*.csv"))]
        if not files:
            st.info("병합할 라인 CSV가 없습니다.")
        else:
            merged = merge_lines(files,dedup)
            st.metric("병합 결과",f"{len(files)}개 파일 → {len(merged)}행")
            table(merged)
            download(merged,"merged_lines.csv","merge_download")
    except (OSError,UnicodeError,ValueError) as error:
        st.error(str(error))

with tabs[2]:
    st.subheader("생산 데이터 검증")
    st.write("필수 열, LOT 누락·중복, 실제 날짜, 양의 검사수량, 양품수량 범위, 설비 누락을 확인합니다. 유형이 비어 있으면 미분류로 포함합니다.")
    a,b,c = st.columns(3)
    a.metric("원본",f"{len(original)}행")
    b.metric("유효",f"{len(valid)}행")
    c.metric("보류",f"{len(rejected)}행")
    if rejected:
        table(rejected)
        download(rejected,"rejected_rows.csv","rejected")
    else:
        st.success("검증 통과: 보류 행이 없습니다.")
    with st.expander("유효 데이터와 계산된 defect_qty",expanded=False):
        table(valid)
        download(valid,"production_with_defect_qty.csv","valid",[*columns] if not valid else None)

with tabs[3]:
    st.subheader("장비별 수율")
    st.caption("장비별 양품 합계 / 검사 합계 × 100. LOT 수율을 단순 평균하지 않습니다.")
    if filtered:
        summary = equipment_summary(filtered)
        chart(summary,{"height":300,"mark":{"type":"bar","color":"#3274be"},"encoding":{
            "x":{"field":"equipment_id","type":"nominal","title":"설비"},
            "y":{"field":"yield_pct","type":"quantitative","title":"수율 (%)","scale":{"domain":[0,100]}},
            "tooltip":[{"field":"equipment_id"},{"field":"yield_pct","format":".2f"},{"field":"inspected_qty"},{"field":"lot_count"}]}})
        table(summary)
        download(summary,"equipment_yield.csv","equipment")
    else:
        st.info("집계할 유효 데이터가 없습니다.")

with tabs[4]:
    st.subheader("생산 LOT와 품질 검사 연결")
    st.caption("품질 행을 모두 유지하는 다대일 연결입니다. 필터에서 제외된 생산 LOT도 미매칭으로 표시됩니다. 검사 건수는 불량수량이 아닙니다.")
    qa_upload = st.file_uploader("quality.csv 업로드 (선택)",type="csv",key="quality")
    try:
        qa_columns, quality = read_csv(qa_upload.getvalue() if qa_upload is not None else (ROOT/"quality.csv").read_bytes())
        joined = join_quality(filtered,qa_columns,quality)
        matched = sum(r["production_matched"] for r in joined)
        a,b,c = st.columns(3)
        a.metric("품질 검사행",len(joined))
        b.metric("연결행",matched)
        c.metric("미매칭",len(joined)-matched)
        table(joined)
        download(joined,"joined_quality.csv","quality_download")
    except (OSError,UnicodeError,ValueError) as error:
        st.error(str(error))

with tabs[5]:
    st.subheader("불량수량 기준 파레토")
    st.caption("defect_qty = 검사수량 − 양품수량. LOT 건수가 아닌 불량수량을 내림차순으로 합산합니다.")
    if filtered:
        summary = pareto_summary(filtered)
        total = sum(r["defect_qty"] for r in summary)
        if total:
            x = {"field":"defect_type","type":"nominal","sort":[r["defect_type"] for r in summary],"title":"불량 유형"}
            chart(summary,{"height":350,"encoding":{"x":x},"layer":[
                {"mark":{"type":"bar","color":"#3274be"},"encoding":{
                    "y":{"field":"defect_qty","type":"quantitative","axis":{"title":"불량수량 (개)"}},
                    "tooltip":[{"field":"defect_type"},{"field":"defect_qty"},{"field":"share_pct","format":".2f"}]}},
                {"mark":{"type":"line","point":True,"color":"#db6536"},"encoding":{
                    "y":{"field":"cumulative_pct","type":"quantitative","scale":{"domain":[0,100]},"axis":{"title":"누적 비중 (%)","orient":"right"}},
                    "tooltip":[{"field":"defect_type"},{"field":"cumulative_pct","format":".2f"}]}}
            ],"resolve":{"scale":{"y":"independent"}}})
            st.success(f"총량 일치: {total:,}개 · 마지막 누적 비중 {summary[-1]['cumulative_pct']:.2f}%")
        else:
            st.info("불량수량이 0이므로 비중과 누적 비중을 정의하지 않습니다.")
        table(summary)
        download(summary,"defect_pareto.csv","pareto")
    else:
        st.info("집계할 유효 데이터가 없습니다.")
    st.info("LOT당 기록된 단일 유형에 해당 LOT의 불량수량 전체를 배정했습니다. 유형별 실제 수량이 없으므로 이 순위는 우선 검토 근거이며 원인 증명이 아닙니다. 누적 100%만으로 원본의 완전성을 보장할 수 없습니다.")

with tabs[6]:
    st.subheader("일별 수율 추이와 검사수량")
    st.caption("trand_chart (1).py와 같은 위아래 구성입니다. 날짜를 실제 날짜형으로 정렬하고 빠진 날을 비워 둡니다.")
    if daily:
        x = {"field":"work_date","type":"temporal","title":"날짜","axis":{"format":"%m-%d"}}
        chart(daily,{"height":280,"mark":{"type":"line","point":True,"color":"#1e2761"},"encoding":{
            "x":x,"y":{"field":"yield_pct","type":"quantitative","title":"수율 (%)","scale":{"domain":[0,100]}},
            "detail":{"field":"segment","type":"nominal"},
            "tooltip":[{"field":"work_date","type":"temporal","format":"%Y-%m-%d"},{"field":"yield_pct","format":".2f"},{"field":"inspected_qty"},{"field":"lot_count"}]}})
        chart(daily,{"height":170,"mark":{"type":"bar","color":"#5b8db8"},"encoding":{
            "x":x,"y":{"field":"inspected_qty","type":"quantitative","title":"검사수량 (개)"},
            "tooltip":[{"field":"work_date","type":"temporal","format":"%Y-%m-%d"},{"field":"inspected_qty"},{"field":"lot_count"}]}})
        missing = [r["work_date"] for r in daily if not r["observed"]]
        st.info("결측 규칙: 최초~최종 관측일 사이의 무기록 날짜는 수량·수율을 null로 두고 선을 끊습니다. 0%로 채우지 않습니다. 기록 없음의 사유와 기간 밖의 날짜는 판단할 수 없습니다.")
        st.write("결측일: " + (", ".join(missing) if missing else "없음"))
        cards_view(cards)
        payload = {"요약카드":cards,"결측규칙":"기간 내 무기록 날짜는 null, 0%로 대체하지 않고 선 끊음.",
                   "수율산식":"양품수량 합계 / 검사수량 합계 × 100",
                   "선택조건":{"생산원본":uploaded.name if uploaded is not None else "production.csv",
                               "설비":equipment,"기간":[d.isoformat() for d in date_range]},
                   "데이터한계":"합성 실습 자료. 제품·설비·검사 구성 미보정. 기록 없음은 미생산을 뜻하지 않음. 원인 증명이 아님.",
                   "AI전달규칙":"이 요약만 사용. 계산값 그대로 인용. 원본·결측·원인을 추정하거나 보충하지 말 것."}
        st.download_button("AI 전달용 요약 6항목 다운로드",json.dumps(payload,ensure_ascii=False,indent=2).encode("utf-8"),
                           "daily_summary_cards.json",mime="application/json")
        with st.expander("일별 집계표"):
            table(daily)
            download(daily,"daily_yield.csv","daily")
    else:
        st.info("집계할 유효 데이터가 없습니다.")

with tabs[7]:
    st.subheader("처리일지와 duration_ms 열")
    st.caption("기존 add_duration.py는 빈 duration_ms 열을 추가합니다. 실제 실행시간을 측정한 값이 아니므로 0으로 채우지 않습니다.")
    try:
        log_columns, logs = read_csv((ROOT/"로그"/"처리일지.csv").read_bytes())
        for row in logs:
            row.setdefault("duration_ms","")
        table(logs)
        download(logs,"처리일지_duration.csv","log",list(dict.fromkeys([*log_columns,"duration_ms"])))
    except (OSError,UnicodeError,ValueError) as error:
        st.error(str(error))

with tabs[8]:
    st.subheader("계산 검증 결과")
    checks = calculation_checks(filtered, len(rejected))
    table(checks)
    scope = {"생산원본": uploaded.name if uploaded is not None else "production.csv",
             "설비":equipment, "기간":[d.isoformat() for d in date_range], "선택_LOT수":len(filtered)}
    with st.expander("검사 실행 기록", expanded=False):
        st.json({"범위":scope, "검증":checks,
                 "실행범위":"현재 선택 데이터의 계산·총량 검증. 브라우저 UI·외부 배포·업무 적합성 검사는 포함하지 않습니다."})
    st.subheader("사람 승인")
    st.info("미확인 — 작은 표 검산, 현장 자료 출처, 유형 귀속 기준에 대한 사람 승인 기록은 없습니다.")
    st.caption("계산 검증 통과는 사람의 승인이나 불량 원인 증명을 뜻하지 않습니다.")
    approval_record = {"선택조건":scope, "계산검증":checks,
                       "사람승인":{"상태":"미확인", "승인자":None, "승인일시":None},
                       "검사한계":"현재 데이터의 계산만 검증. 화면·외부 배포·업무 적합성은 별도 확인 필요."}
    st.download_button("검증·승인 상태 다운로드",json.dumps(approval_record,ensure_ascii=False,indent=2).encode("utf-8"),
                       "verification_status.json",mime="application/json")

st.divider()
st.caption("해석 범위: 수량·수율·분모를 함께 확인하세요. 이 자료는 합성 실습 데이터이며, 순위와 추세는 원인을 증명하지 않습니다.")
