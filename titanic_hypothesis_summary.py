# -*- coding: utf-8 -*-
"""가설 H1~H6을 한 장으로 압축한 요약 차트 (오즈비 포레스트).

각 행은 해당 가설의 검정 모형에서 나온 오즈비와 95% 신뢰구간이다.
  - 기준선 1의 오른쪽 = 생존에 유리, 왼쪽 = 불리
  - 파랑 = 유의(p<0.05), 회색 = 비유의 / 판정 불가
  - 가로 길이(신뢰구간 폭)가 곧 그 추정치의 불확실성

사용법:
    python titanic_hypothesis_summary.py           # -> titanic_hypothesis_summary.png
    python titanic_hypothesis_summary.py --dark    # -> titanic_hypothesis_summary_dark.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

from titanic_hypotheses import logit, lr_test, prepare
from titanic_survival import DARK, LIGHT, load_data, setup_font

HERE = Path(__file__).resolve().parent
EXTRA_LIGHT = {"good": "#0ca30c"}
EXTRA_DARK = {"good": "#0ca30c"}


# ── 행 데이터 만들기 ─────────────────────────────────────────────────────────
def effect(res, key: str) -> tuple[float, float, float, float]:
    ci = res.conf_int()
    return (
        float(np.exp(res.params[key])),
        float(np.exp(ci.loc[key, 0])),
        float(np.exp(ci.loc[key, 1])),
        float(res.pvalues[key]),
    )


def build_rows(d: pd.DataFrame) -> list[dict]:
    """(가설, 항목, 오즈비, CI, p) 행 목록. 각 행은 그 가설의 검정 모형에서 나온다."""
    y, base = d["Survived"], d[["여성", "p2", "p3"]]
    m_base = logit(y, base)
    rows: list[dict] = []

    # H1 — 성별·등급 동시 투입
    od, lo, hi, p = effect(m_base, "여성")
    rows.append({"tag": "H1", "label": "여성 (vs 남성)", "od": od, "lo": lo, "hi": hi, "p": p})

    # H2 — 성별로 층화했을 때 '3등급 페널티'가 서로 다른가
    inter = base.assign(**{"f2": d["여성"] * d["p2"], "f3": d["여성"] * d["p3"]})
    t2 = lr_test(logit(y, inter), m_base)
    for sex in ("여성", "남성"):
        sub = d[d["성별"] == sex]
        m = logit(sub["Survived"], sub[["p2", "p3"]])
        od, lo, hi, p = effect(m, "p3")
        rows.append({"tag": "H2", "label": f"3등급 · {sex} 내", "od": od, "lo": lo, "hi": hi, "p": p})
    rows[-2]["tag_note"] = f"교호작용 LR p={t2['p']:.1e}"

    # H3 — 검정 자체가 불가능
    cov3 = d[d["Pclass"] == 3]["선실기록"].mean() * 100
    rows.append(
        {
            "tag": "H3",
            "label": "갑판(선실) 위치",
            "na": True,
            "na_text": f"3등급 선실 기재율 {cov3:.1f}% → 검정 불가",
        }
    )

    # H4 — 가족 규모 (성별·등급 통제)
    fam = pd.get_dummies(d["가족규모"], prefix="fam", drop_first=True).astype(int)
    m4 = logit(y, pd.concat([base, fam], axis=1))
    for key, label in (("fam_소가족(2-4인)", "소가족 2-4인 (vs 단독)"), ("fam_대가족(5인+)", "대가족 5인+ (vs 단독)")):
        od, lo, hi, p = effect(m4, key)
        rows.append({"tag": "H4", "label": label, "od": od, "lo": lo, "hi": hi, "p": p})

    # H5 — 승선항 (성별·등급 통제)
    e = d.dropna(subset=["Embarked"])
    emb = pd.get_dummies(e["Embarked"], prefix="emb", drop_first=True).astype(int)
    m5 = logit(e["Survived"], pd.concat([e[["여성", "p2", "p3"]], emb], axis=1))
    for key, label in (("emb_Q", "Q 퀸스타운 (vs C)"), ("emb_S", "S 사우샘프턴 (vs C)")):
        od, lo, hi, p = effect(m5, key)
        rows.append({"tag": "H5", "label": label, "od": od, "lo": lo, "hi": hi, "p": p})

    # H6 — 결측 (성별·등급 통제)
    for col, label in (("선실기록", "선실 기재됨 (vs 미기재)"), ("나이결측", "나이 미기재 (vs 기재됨)")):
        m = logit(y, base.assign(**{col: d[col].astype(int)}))
        od, lo, hi, p = effect(m, col)
        rows.append({"tag": "H6", "label": label, "od": od, "lo": lo, "hi": hi, "p": p})

    return rows


VERDICTS = {
    "H1": ("지지", True),
    "H2": ("지지", True),
    "H3": ("검증 불가", False),
    "H4": ("지지", True),
    "H5": ("지지", True),
    "H6": ("지지", True),
}
TAG_TITLES = {
    "H1": "성별이 1차 관문",
    "H2": "등급×성별 교호작용",
    "H3": "갑판 위치 가설",
    "H4": "가족 규모 역U자",
    "H5": "승선항 독립 효과",
    "H6": "결측의 비무작위성",
}


# ── 그리기 ───────────────────────────────────────────────────────────────────
def build_figure(dark: bool) -> plt.Figure:
    c = dict(DARK if dark else LIGHT, **(EXTRA_DARK if dark else EXTRA_LIGHT))
    d = prepare(load_data())
    rows = build_rows(d)

    fig = plt.figure(figsize=(13.6, 9.4), dpi=140, facecolor=c["surface"])
    ax = fig.add_axes([0.335, 0.115, 0.395, 0.70])
    ax.set_facecolor(c["surface"])

    finite = [r for r in rows if not r.get("na")]
    xmin = max(min(r["lo"] for r in finite) * 0.55, 0.005)
    xmax = min(max(r["hi"] for r in finite) * 1.8, 200)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=c["muted"], length=0, labelsize=10)
    ax.axvline(1, color=c["axis"], linewidth=1.0, zorder=1)

    ys = list(range(len(rows) - 1, -1, -1))  # 첫 행이 맨 위
    for y, r in zip(ys, rows):
        if r.get("na"):
            ax.text(
                np.sqrt(xmin * xmax), y, r["na_text"],
                ha="center", va="center", color=c["muted"], fontsize=10, style="italic",
            )
            continue
        sig = r["p"] < 0.05
        color = c["alive"] if sig else c["muted"]
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(
            [r["od"]], [y], "o", markersize=9, color=color,
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4,
        )

    # 가설 그룹 구분선
    for k in range(1, len(rows)):
        if rows[k]["tag"] != rows[k - 1]["tag"]:
            ax.axhline(ys[k] + 0.5, color=c["grid"], linewidth=0.8, zorder=0, xmin=-0.76, clip_on=False)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=11, color=c["ink"])
    ax.tick_params(axis="y", labelcolor=c["ink"], pad=8)
    ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("오즈비 (로그 눈금) — 기준선 1보다 오른쪽이면 생존에 유리", color=c["ink2"], fontsize=10.5, labelpad=12)

    # 왼쪽: 가설 태그 / 오른쪽: 수치와 판정
    seen: set[str] = set()
    for y, r in zip(ys, rows):
        tag = r["tag"]
        if tag not in seen:
            seen.add(tag)
            judgement, ok = VERDICTS[tag]
            ax.text(
                -0.735, y, tag, transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=c["ink"], fontsize=12, fontweight="bold", clip_on=False,
            )
            ax.text(
                -0.672, y, TAG_TITLES[tag], transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=c["ink2"], fontsize=10, clip_on=False,
            )
            ax.text(
                1.43, y, f"● {judgement}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=c["good"] if ok else c["muted"],
                fontsize=10.5, fontweight="bold", clip_on=False,
            )
        if r.get("na"):
            continue
        sig = r["p"] < 0.05
        ax.text(
            1.06, y, f"{r['od']:.2f}", transform=ax.get_yaxis_transform(),
            ha="right", va="center", color=c["ink"] if sig else c["muted"],
            fontsize=11, fontweight="bold" if sig else "normal", clip_on=False,
        )
        ax.text(
            1.10, y, f"[{r['lo']:.2f}, {r['hi']:.2f}]" + ("" if sig else "  n.s."),
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", color=c["ink2"] if sig else c["muted"], fontsize=9.5, clip_on=False,
        )

    # 열 머리글
    for x, text, ha in ((1.06, "오즈비", "right"), (1.10, "95% 신뢰구간", "left"), (1.43, "판정", "left")):
        ax.text(
            x, len(rows) - 0.15, text, transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", color=c["muted"], fontsize=9.5, clip_on=False,
        )

    fig.text(0.05, 0.955, "무엇이 생존을 갈랐나 — 가설 6개 요약", color=c["ink"], fontsize=23, fontweight="bold", va="top")
    fig.text(
        0.05, 0.912,
        "탑승객 891명 · 각 항은 해당 가설의 검정 모형에서 나온 오즈비(H2는 성별 층화, H4~H6은 성별·등급 통제)",
        color=c["ink2"], fontsize=11.5, va="top",
    )
    fig.text(
        0.05, 0.884,
        "가로선 = 95% 신뢰구간 · 파랑 = 유의(p<0.05) · 회색 = 비유의(n.s.) 또는 판정 불가",
        color=c["muted"], fontsize=10, va="top",
    )
    fig.text(
        0.05, 0.045,
        "읽는 법: 1에서 멀수록 효과가 크고, 가로선이 짧을수록 추정이 확실하다. 가로선이 기준선 1을 물면 유의하지 않다.",
        color=c["ink2"], fontsize=10.5,
    )
    fig.text(
        0.05, 0.018,
        "데이터: titanic.csv · 연관성이며 인과가 아님 · p값은 다중비교 보정 전 · 상세 근거는 리포트 파일 kk",
        color=c["muted"], fontsize=9.5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="가설 요약 1장 차트")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로")
    args = parser.parse_args()

    setup_font()
    c = DARK if args.dark else LIGHT
    fig = build_figure(dark=args.dark)
    out = args.out or HERE / (
        "titanic_hypothesis_summary_dark.png" if args.dark else "titanic_hypothesis_summary.png"
    )
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
