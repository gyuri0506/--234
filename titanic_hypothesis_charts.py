# -*- coding: utf-8 -*-
"""가설 H1~H6의 검정 결과를 시각화한다 (kk 리포트의 그림판).

가설마다 데이터 성격에 맞는 형태를 쓴다.
  H1 오즈비 포레스트    - 통제 후 효과 크기와 불확실성을 한 축에서
  H2 교호작용 선그래프  - 두 선의 '기울기 차이'가 곧 가설의 내용
  H3 검증 가능 범위 막대 - 데이터가 없다는 사실 자체를 보여줌
  H4 구간별 생존율 + CI - 역U자 형태 확인
  H5 층화 점도표 + CI   - 등급을 고정한 뒤에도 남는 격차
  H6 덤벨               - 결측 유무에 따른 생존율 간격

사용법:
    python titanic_hypothesis_charts.py           # -> titanic_hypothesis_charts.png
    python titanic_hypothesis_charts.py --dark    # -> titanic_hypothesis_charts_dark.png
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

from titanic_hypotheses import logit, lr_test, prepare, wilson
from titanic_survival import DARK, LIGHT, _px_to_data, load_data, rounded_bar, setup_font

HERE = Path(__file__).resolve().parent

# 문서화된 팔레트에서 이 그림이 추가로 쓰는 슬롯만 가져온다.
EXTRA_LIGHT = {"soft": "#9ec5f4", "series2": "#eb6834", "good": "#0ca30c"}
EXTRA_DARK = {"soft": "#256abf", "series2": "#d95926", "good": "#0ca30c"}

MARGIN_LEFT = 0.095  # 패널과 머리글의 공통 왼쪽 정렬선


def pt2frac(fig: plt.Figure, pt: float) -> float:
    """포인트 단위 여백을 figure 높이 기준 비율로."""
    return pt / (fig.get_figheight() * 72)


def head(fig: plt.Figure, cell, tag: str, title: str, note: str, judgement: str, ok: bool, c: dict) -> None:
    """패널 제목 + 판정 칩. 판정은 색만이 아니라 '●+글자'로 표시한다."""
    fig.text(
        cell.x0,
        cell.y1 + pt2frac(fig, 34),
        f"{tag}. {title}",
        color=c["ink"],
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )
    fig.text(cell.x0, cell.y1 + pt2frac(fig, 14), note, color=c["ink2"], fontsize=9.5, va="bottom")
    fig.text(
        cell.x1,
        cell.y1 + pt2frac(fig, 34),
        f"● {judgement}",
        color=c["good"] if ok else c["muted"],
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def style(ax: plt.Axes, c: dict, grid_axis: str | None = None) -> None:
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=c["grid"], linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=c["muted"], length=0, labelsize=9.5)


def pct_axis(ax: plt.Axes, axis: str = "y") -> None:
    ticks = [0, 25, 50, 75, 100]
    labels = ["0", "25", "50", "75", "100%"]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


# ── H1: 오즈비 포레스트 ──────────────────────────────────────────────────────
def p_h1(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    res = logit(d["Survived"], d[["여성", "p2", "p3"]])
    ci = res.conf_int()
    items = [("여성", "여성"), ("2등급", "p2"), ("3등급", "p3")]

    ax.set_xscale("log")
    ax.set_xlim(0.06, 45)
    ax.set_ylim(-0.6, len(items) - 0.4)
    style(ax, c, grid_axis="x")
    ax.axvline(1, color=c["axis"], linewidth=0.9, zorder=1)

    for k, (label, key) in enumerate(items):
        y = len(items) - 1 - k
        lo, hi, od = np.exp(ci.loc[key, 0]), np.exp(ci.loc[key, 1]), np.exp(res.params[key])
        ax.plot([lo, hi], [y, y], color=c["alive"], linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(
            [od], [y], "o", markersize=9, color=c["alive"],
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4,
        )
        ax.text(hi * 1.35, y, f"{od:.2f}배", va="center", ha="left", color=c["ink"], fontsize=10.5, fontweight="bold")

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for lab, _ in items][::-1], fontsize=10.5, color=c["ink"])
    ax.tick_params(axis="y", labelcolor=c["ink"])
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.text(
        0.99, 0.04,
        "기준선 1의 왼쪽 = 생존에 불리 · 오른쪽 = 유리\n가로선 = 95% 신뢰구간",
        transform=ax.transAxes, ha="right", va="bottom", color=c["muted"], fontsize=9, linespacing=1.7,
    )
    head(
        ax.figure, cell, "H1", "성별이 1차 관문, 등급은 2차",
        f"로지스틱 회귀 오즈비 (기준: 남성·1등급) · 여성 p={res.pvalues['여성']:.1e}",
        "지지", True, c,
    )


# ── H2: 교호작용 선그래프 ────────────────────────────────────────────────────
def p_h2(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    base = d[["여성", "p2", "p3"]]
    inter = base.assign(**{"f2": d["여성"] * d["p2"], "f3": d["여성"] * d["p3"]})
    t = lr_test(logit(d["Survived"], inter), logit(d["Survived"], base))

    xs = np.array([0, 1, 2])
    ax.set_xlim(-0.45, 2.75)
    ax.set_ylim(0, 105)
    style(ax, c, grid_axis="y")

    for sex, color in (("여성", c["alive"]), ("남성", c["series2"])):
        sub = d[d["성별"] == sex]
        rates = [sub[sub["Pclass"] == p]["Survived"].mean() * 100 for p in (1, 2, 3)]
        ax.plot(xs, rates, color=color, linewidth=2, zorder=3, label=sex)
        ax.plot(
            xs, rates, "o", markersize=9, color=color,
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4,
        )
        ax.text(
            xs[-1] + 0.12, rates[-1], f"{sex} {rates[-1]:.1f}%",
            va="center", ha="left", color=color, fontsize=10.5, fontweight="bold",
        )
        ax.text(
            xs[0] - 0.1, rates[0], f"{rates[0]:.1f}%",
            va="center", ha="right", color=c["ink2"], fontsize=10,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["1등급", "2등급", "3등급"], fontsize=10.5, color=c["ink"])
    ax.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax, "y")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower left", handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    ax.annotate(
        "두 선이 평행하지 않음 = 교호작용",
        xy=(1.0, 54), xytext=(0.2, 78), color=c["ink2"], fontsize=9.5,
        arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9},
    )
    head(
        ax.figure, cell, "H2", "등급 효과가 성별에 따라 다르다",
        f"성별 내 등급별 생존율 · 우도비 검정 LR={t['lr']:.1f}, p={t['p']:.1e}",
        "지지", True, c,
    )


# ── H3: 검증 가능 범위 ──────────────────────────────────────────────────────
def p_h3(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    cov = d.groupby("등급", observed=True)["선실기록"].agg(n="size", k="sum")
    cov["rate"] = cov["k"] / cov["n"] * 100

    xs = np.arange(len(cov))
    ax.set_xlim(-0.6, len(cov) - 0.4)
    ax.set_ylim(0, 105)
    style(ax, c, grid_axis="y")
    rx, ry = _px_to_data(ax, 4)

    for x, (idx, row) in zip(xs, cov.iterrows()):
        rounded_bar(ax, x - 0.3, 0, x + 0.3, row["rate"], rx, ry, ("tl", "tr"), c["alive"])
        ax.text(
            x, row["rate"] + 3, f"{row['rate']:.1f}%",
            ha="center", va="bottom", color=c["ink"], fontsize=10.5, fontweight="bold",
        )
        ax.text(
            x, row["rate"] + 11, f"{int(row['k'])}/{int(row['n'])}명",
            ha="center", va="bottom", color=c["muted"], fontsize=9,
        )

    ax.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(cov.index), fontsize=10.5, color=c["ink"])
    ax.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax, "y")
    ax.annotate(
        "검증 대상인 3등급의 위치 정보가\n사실상 없음 → 이 데이터로 판정 불가",
        xy=(2, 4), xytext=(1.12, 42), color=c["ink"], fontsize=9.5, linespacing=1.6,
        arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9},
    )
    head(
        ax.figure, cell, "H3", "갑판 위치로 3등급 여성 급락을 설명",
        "등급별 선실(Cabin) 기재율 = 이 가설을 검증할 수 있는 범위",
        "검증 불가", False, c,
    )


# ── H4: 구간별 생존율 + 신뢰구간 ─────────────────────────────────────────────
def p_h4(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    g = d.groupby("가족규모", observed=True)["Survived"].agg(n="size", s="sum")
    base = d[["여성", "p2", "p3"]]
    fam = pd.get_dummies(d["가족규모"], prefix="fam", drop_first=True).astype(int)
    t = lr_test(logit(d["Survived"], pd.concat([base, fam], axis=1)), logit(d["Survived"], base))

    xs = np.arange(len(g))
    ax.set_xlim(-0.6, len(g) - 0.4)
    ax.set_ylim(0, 105)
    style(ax, c, grid_axis="y")
    rx, ry = _px_to_data(ax, 4)

    peak = g.index[(g["s"] / g["n"]).argmax()]
    for x, (idx, row) in zip(xs, g.iterrows()):
        rate = row["s"] / row["n"] * 100
        lo, hi = wilson(int(row["s"]), int(row["n"]))
        # 강조: 이야기의 주인공(생존율 최고 구간)만 파랑, 나머지는 비강조 그레이
        rounded_bar(ax, x - 0.3, 0, x + 0.3, rate, rx, ry, ("tl", "tr"), c["alive"] if idx == peak else c["dead"])
        ax.plot([x, x], [lo, hi], color=c["ink2"], linewidth=1.6, solid_capstyle="round", zorder=5)
        ax.text(x, hi + 3, f"{rate:.1f}%", ha="center", va="bottom", color=c["ink"], fontsize=10.5, fontweight="bold")

    ax.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{i}\nn={int(n)}" for i, n in zip(g.index, g["n"])],
        fontsize=10, color=c["ink"], linespacing=1.6,
    )
    ax.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax, "y")
    w3 = d[(d["성별"] == "여성") & (d["Pclass"] == 3)].groupby("가족규모", observed=True)["Survived"].mean() * 100
    ax.text(
        0.025, 0.98,
        "세로선 = Wilson 95% 신뢰구간\n"
        f"3등급 여성 안에서도: 단독 {w3.iloc[0]:.1f}% · 소가족 {w3.iloc[1]:.1f}% · 대가족 {w3.iloc[2]:.1f}%",
        transform=ax.transAxes, va="top", ha="left", color=c["ink2"], fontsize=9.5, linespacing=1.7,
    )
    head(
        ax.figure, cell, "H4", "가족 규모가 생존을 좌우 (역U자)",
        f"성별·등급 통제 후에도 유효 · LR={t['lr']:.1f}, p={t['p']:.1e}",
        "지지", True, c,
    )


# ── H5: 층화 점도표 ─────────────────────────────────────────────────────────
def p_h5(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    e = d.dropna(subset=["Embarked"])
    base = e[["여성", "p2", "p3"]]
    emb = pd.get_dummies(e["Embarked"], prefix="emb", drop_first=True).astype(int)
    t = lr_test(logit(e["Survived"], pd.concat([base, emb], axis=1)), logit(e["Survived"], base))

    sub = e[(e["성별"] == "여성") & (e["Pclass"] == 3)]
    g = sub.groupby("Embarked", observed=True)["Survived"].agg(n="size", s="sum")
    names = {"C": "C 셰르부르", "Q": "Q 퀸스타운", "S": "S 사우샘프턴"}

    ys = np.arange(len(g))[::-1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(g) - 0.4)
    style(ax, c, grid_axis="x")

    for y, (idx, row) in zip(ys, g.iterrows()):
        rate = row["s"] / row["n"] * 100
        lo, hi = wilson(int(row["s"]), int(row["n"]))
        ax.plot([lo, hi], [y, y], color=c["alive"], linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(
            [rate], [y], "o", markersize=9, color=c["alive"],
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4,
        )
        ax.text(hi + 2.5, y, f"{rate:.1f}%", va="center", ha="left", color=c["ink"], fontsize=10.5, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{names[i]}\nn={int(n)}" for i, n in zip(g.index, g["n"])],
        fontsize=10, color=c["ink"], linespacing=1.6,
    )
    ax.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax, "x")
    ax.text(
        0.99, 0.06,
        "가로선 = Wilson 95% 신뢰구간\n(표본이 작아 폭이 넓음)",
        transform=ax.transAxes, va="bottom", ha="right", color=c["muted"], fontsize=9, linespacing=1.7,
    )
    head(
        ax.figure, cell, "H5", "승선항은 등급의 대리변수가 아니다",
        f"'3등급 여성'으로 고정해도 남는 격차 · 통제 후 p={t['p']:.3f} (유의한 대비는 S vs C)",
        "지지", True, c,
    )


# ── H6: 덤벨 ────────────────────────────────────────────────────────────────
def p_h6(ax: plt.Axes, cell, d: pd.DataFrame, c: dict) -> None:
    base = d[["여성", "p2", "p3"]]
    specs = []
    for col, label in (("선실기록", "선실(Cabin)"), ("나이결측", "나이(Age)")):
        t = lr_test(logit(d["Survived"], base.assign(**{col: d[col].astype(int)})), logit(d["Survived"], base))
        flag = d[col] if col == "선실기록" else ~d[col]  # 둘 다 '기재됨=True'로 맞춘다
        specs.append((label, d[flag]["Survived"].mean() * 100, d[~flag]["Survived"].mean() * 100, t))

    ys = np.arange(len(specs))[::-1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, len(specs) - 0.25)
    style(ax, c, grid_axis="x")

    for y, (label, ry, rn, t) in zip(ys, specs):
        ax.plot([rn, ry], [y, y], color=c["axis"], linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([rn], [y], "o", markersize=9, color=c["soft"], markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax.plot([ry], [y], "o", markersize=9, color=c["alive"], markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax.text(min(rn, ry) - 2.5, y, f"{min(rn, ry):.1f}%", va="center", ha="right", color=c["ink2"], fontsize=10)
        ax.text(
            max(rn, ry) + 2.5, y, f"{max(rn, ry):.1f}%",
            va="center", ha="left", color=c["ink"], fontsize=10.5, fontweight="bold",
        )
        tail = "통제 후 효과 소멸 → MAR" if t["p"] >= 0.05 else "통제 후에도 유의 → MNAR 의심"
        ax.text(
            min(rn, ry), y - 0.3,
            f"성별·등급 통제 후 p={t['p']:.3f} · {tail}",
            va="top", ha="left", color=c["muted"], fontsize=9,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in specs], fontsize=10.5, color=c["ink"])
    ax.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax, "x")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=c["alive"], label="기재됨"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=c["soft"], label="미기재"),
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper right", ncol=2, handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    head(
        ax.figure, cell, "H6", "결측 자체가 생존과 상관있다",
        "기재 여부에 따른 생존율 간격 · 두 결측의 성격이 서로 다름",
        "지지", True, c,
    )


# ── 그림 조립 ────────────────────────────────────────────────────────────────
def build_figure(dark: bool) -> plt.Figure:
    c = dict(DARK if dark else LIGHT, **(EXTRA_DARK if dark else EXTRA_LIGHT))
    d = prepare(load_data())

    fig = plt.figure(figsize=(14.4, 16.2), dpi=120, facecolor=c["surface"])
    gs = fig.add_gridspec(3, 2, left=MARGIN_LEFT, right=0.955, top=0.845, bottom=0.045, wspace=0.30, hspace=0.42)

    fig.text(
        MARGIN_LEFT, 0.965, "타이타닉 생존 가설 검정 — 시각 요약",
        color=c["ink"], fontsize=22, fontweight="bold", va="top",
    )
    fig.text(
        MARGIN_LEFT, 0.9385,
        "탑승객 891명 · 가설 6개를 카이제곱 · 로지스틱 회귀(우도비 검정) · Wilson 95% 신뢰구간으로 검정",
        color=c["ink2"], fontsize=11.5, va="top",
    )
    fig.text(
        MARGIN_LEFT, 0.915,
        "판정은 통계적 유의성이며 인과가 아님 · p값은 다중비교 보정 전 값 (Bonferroni 기준 α=0.0083)",
        color=c["muted"], fontsize=10, va="top",
    )

    for k, fn in enumerate((p_h1, p_h2, p_h3, p_h4, p_h5, p_h6)):
        row, col = divmod(k, 2)
        ax = fig.add_subplot(gs[row, col])
        fn(ax, gs[row, col].get_position(fig), d, c)

    fig.text(
        MARGIN_LEFT, 0.012,
        "데이터: titanic.csv · 수치 상세와 판정 근거는 리포트 파일 kk 참조",
        color=c["muted"], fontsize=9.5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="가설 검정 결과 시각화")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로")
    args = parser.parse_args()

    setup_font()
    c = DARK if args.dark else LIGHT
    fig = build_figure(dark=args.dark)
    out = args.out or HERE / (
        "titanic_hypothesis_charts_dark.png" if args.dark else "titanic_hypothesis_charts.png"
    )
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
