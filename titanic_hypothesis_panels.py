# -*- coding: utf-8 -*-
"""가설 H1~H6을 '가설당 한 장'으로, 각 가설의 논증 구조에 맞춰 시각화한다.

titanic_hypothesis_charts.py 가 6개를 한 장에 요약한 그림이라면, 이 스크립트는
가설마다 전용 지면을 주고 '근거의 순서'까지 그림으로 옮긴다.

  H1  효과 크기 막대 → 오즈비 포레스트 → 호칭별 점도표   (3단 논증: 크기·통제·분해)
  H2  교호작용 선그래프 + 성별 낙폭 막대                 (두 선의 기울기 차이가 곧 가설)
  H3  선실 기재율 막대 + 1등급 내부 갑판 점도표           ('검증 가능 범위'와 그 안의 결과)
  H4  가족수별 생존율 막대(+CI) + 3등급 여성 층화 점도표  (역U자와 교란 통제)
  H5  승선항별 등급 구성 누적막대 + 3등급 여성 점도표     (대리변수 의심 → 고정 후 잔여 격차)
  H6  덤벨 + 등급별 미기재율 그룹막대                     (결측 격차와 그 출처)

사용법:
    python titanic_hypothesis_panels.py            # 라이트 6장 -> hypothesis_panels/H1.png ...
    python titanic_hypothesis_panels.py --dark     # 다크 6장   -> hypothesis_panels/H1_dark.png ...
    python titanic_hypothesis_panels.py --both     # 라이트+다크 12장
    python titanic_hypothesis_panels.py --only H3  # 특정 가설만
    python titanic_hypothesis_panels.py --table    # 각 그림의 수치를 콘솔 표로도 출력
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

from titanic_hypotheses import chi2_v, logit, lr_test, prepare, wilson
from titanic_survival import DARK, LIGHT, _px_to_data, load_data, rounded_bar, setup_font

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "hypothesis_panels"

# 문서화된 기준 팔레트에서 이 그림들이 추가로 쓰는 슬롯만 가져온다.
#   series2 : 범주 슬롯2(주황) — 슬롯1(파랑)과의 인접쌍은 검증된 조합
#   ramp    : 순서형(등급) 3단계 블루 램프. 라이트는 250 이상, 다크는 550 이하로 제한
#   good    : 상태색 good(판정 칩 전용, 시리즈 색으로는 쓰지 않음)
EXTRA_LIGHT = {"series2": "#eb6834", "good": "#0ca30c", "ramp": ("#2a78d6", "#5598e7", "#86b6ef")}
EXTRA_DARK = {"series2": "#d95926", "good": "#0ca30c", "ramp": ("#1c5cab", "#3987e5", "#86b6ef")}

FIGH = 8.4  # 모든 장의 높이를 맞춘다(연달아 넘겨 볼 때 머리글 위치가 흔들리지 않게)
R = 0.975
TOP, BOTTOM = 0.775, 0.215  # 패널 영역 — 아래쪽은 축 레이블·주석·꼬리말 자리
FOOT_Y = 0.075


# ── 공통 골격 ────────────────────────────────────────────────────────────────
def frame(c: dict, width: float, left: float, tag: str, title: str, lead: str,
          verdict: str, ok: bool, foot: str) -> plt.Figure:
    """머리글(가설 번호·제목·판정 칩)과 꼬리말을 얹은 빈 지면."""
    fig = plt.figure(figsize=(width, FIGH), dpi=130, facecolor=c["surface"])
    fig.text(left, 0.955, f"{tag}. {title}", color=c["ink"], fontsize=20, fontweight="bold", va="top")
    fig.text(left, 0.893, lead, color=c["ink2"], fontsize=11.5, va="top", linespacing=1.6)
    # 판정은 색만이 아니라 '●+글자'로 — 색각 환경에서도 읽히도록
    fig.text(
        R, 0.952, f"● {verdict}",
        color=c["good"] if ok else c["muted"], fontsize=13, fontweight="bold", va="top", ha="right",
    )
    fig.text(left, FOOT_Y, foot, color=c["muted"], fontsize=9.5, va="top", linespacing=1.8)
    return fig


def cap(fig: plt.Figure, cell, text: str, c: dict) -> None:
    """패널 캡션(그리드 칸 기준 정렬)."""
    fig.text(cell.x0, cell.y1 + 0.032, text, color=c["ink"], fontsize=11, fontweight="bold", va="bottom")


def note(fig: plt.Figure, cell, text: str, c: dict, dy: float = 0.07) -> None:
    """패널 아래 주석. 축 안에 넣으면 눈금 레이블과 부딪히므로 축 바깥에 둔다."""
    fig.text(cell.x0, cell.y0 - dy, text, color=c["ink2"], fontsize=9.5, va="top", linespacing=1.8)


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
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(labels)


def dot_ci(ax: plt.Axes, y: float, rate: float, lo: float, hi: float, c: dict, color: str | None = None) -> None:
    """가로 점도표 한 줄: 신뢰구간 선 + 9px 마커(서피스 링 2px) + 직접 레이블."""
    color = color or c["alive"]
    ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=3)
    ax.plot([rate], [y], "o", markersize=9, color=color,
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
    ax.text(hi + 2.2, y, f"{rate:.1f}%", va="center", ha="left", color=c["ink"], fontsize=10.5, fontweight="bold")


def vbar_ci(ax: plt.Axes, x: float, rate: float, lo: float, hi: float, c: dict, color: str, half: float = 0.32) -> None:
    """세로 막대 + Wilson 신뢰구간 수염 + 값 레이블."""
    rx, ry = _px_to_data(ax, 4)
    rounded_bar(ax, x - half, 0, x + half, rate, rx, ry, ("tl", "tr"), color)
    ax.plot([x, x], [lo, hi], color=c["ink2"], linewidth=1.6, solid_capstyle="round", zorder=5)
    ax.text(x, hi + 2.5, f"{rate:.0f}%", ha="center", va="bottom", color=c["ink"], fontsize=10, fontweight="bold")


def rates(df: pd.DataFrame, by, order=None) -> pd.DataFrame:
    """집단별 n·생존수·생존율·Wilson 95% CI."""
    g = df.groupby(by, observed=True)["Survived"].agg(n="size", k="sum")
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g["rate"] = g["k"] / g["n"] * 100
    ci = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["lo"] = [a for a, _ in ci]
    g["hi"] = [b for _, b in ci]
    return g


def show(name: str, g: pd.DataFrame) -> None:
    """색 없이도 값을 읽을 수 있는 대체 경로(콘솔 표)."""
    print(f"\n  [{name}]")
    print(f"  {'집단':<16}{'n':>6}{'생존':>6}{'생존율':>9}{'95% CI':>18}")
    print("  " + "-" * 55)
    for idx, r in g.iterrows():
        ci = f"{r['lo']:.1f} – {r['hi']:.1f}%"
        print(f"  {str(idx):<16}{int(r['n']):>6}{int(r['k']):>6}{r['rate']:>8.1f}%{ci:>18}")


# ── H1 ───────────────────────────────────────────────────────────────────────
def fig_h1(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """근거 3단: ① 연관성 크기 ② 상호 통제 ③ 남성 내부 분해."""
    sex, cls = chi2_v(d, "Sex"), chi2_v(d, "Pclass")
    res = logit(d["Survived"], d[["여성", "p2", "p3"]])
    ci = res.conf_int()
    tit = rates(d, "호칭군").sort_values("rate", ascending=False)

    left = 0.105
    fig = frame(
        c, 13.8, left, "H1", "성별이 1차 관문이고, 객실 등급은 그 다음 필터다",
        "같은 결론을 세 가지 방법으로 확인한다 — 연관성의 크기, 서로 통제한 뒤의 효과, 남성 내부를 쪼갠 결과",
        "지지", True,
        f"카이제곱: 성별 χ²={sex['chi2']:.1f}, p={sex['p']:.1e} · 등급 χ²={cls['chi2']:.1f}, p={cls['p']:.1e}  |  "
        f"로지스틱 회귀(n={len(d)}): 여성 p={res.pvalues['여성']:.1e}\n"
        "데이터: titanic.csv 891명 · 판정은 통계적 유의성이며 인과가 아님",
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.4], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.55)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ① 효과 크기 — 같은 척도(Cramér's V)에서 두 변수를 직접 비교
    cell = gs[0, 0].get_position(fig)
    items = [("성별", sex["v"], c["alive"]), ("객실 등급", cls["v"], c["dead"])]
    ax1.set_xlim(0, 0.68)
    ax1.set_ylim(-0.5, 1.5)
    style(ax1, c, grid_axis="x")
    rx, ry = _px_to_data(ax1, 4)
    for k, (label, v, color) in enumerate(items):
        y = 1 - k
        rounded_bar(ax1, 0, y - 0.24, v, y + 0.24, rx, ry, ("br", "tr"), color)
        ax1.text(v + 0.02, y, f"{v:.3f}", va="center", ha="left", color=c["ink"], fontsize=11.5, fontweight="bold")
    ax1.axvline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax1.set_yticks([1, 0])
    ax1.set_yticklabels([i[0] for i in items], fontsize=11, color=c["ink"])
    ax1.tick_params(axis="y", labelcolor=c["ink"])
    ax1.set_xticks([0, 0.2, 0.4, 0.6])
    cap(fig, cell, "① 어느 쪽이 더 센가 — Cramér's V", c)
    note(fig, cell, f"0 = 무관 · 1 = 완전 결정\n성별 쪽이 {sex['v'] / cls['v']:.1f}배 강하다", c)

    # ② 상호 통제 — 오즈비 포레스트(로그축, 기준선 1)
    cell = gs[0, 1].get_position(fig)
    rows = [("여성", "여성"), ("2등급", "p2"), ("3등급", "p3")]
    ax2.set_xscale("log")
    ax2.set_xlim(0.06, 60)
    ax2.set_ylim(-0.5, 2.5)
    style(ax2, c, grid_axis="x")
    ax2.axvline(1, color=c["axis"], linewidth=1.0, zorder=1)
    for k, (label, key) in enumerate(rows):
        y = 2 - k
        lo, hi, od = np.exp(ci.loc[key, 0]), np.exp(ci.loc[key, 1]), np.exp(res.params[key])
        ax2.plot([lo, hi], [y, y], color=c["alive"], linewidth=2, solid_capstyle="round", zorder=3)
        ax2.plot([od], [y], "o", markersize=9, color=c["alive"],
                 markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax2.text(hi * 1.3, y, f"{od:.2f}배", va="center", ha="left", color=c["ink"], fontsize=11, fontweight="bold")
    ax2.set_yticks([2, 1, 0])
    ax2.set_yticklabels([r[0] for r in rows], fontsize=11, color=c["ink"])
    ax2.tick_params(axis="y", labelcolor=c["ink"])
    ax2.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax2.set_xticklabels(["0.1", "1", "10"])
    cap(fig, cell, "② 서로 통제해도 남는가 — 오즈비", c)
    note(fig, cell, "기준선 1의 왼쪽 = 생존에 불리 · 오른쪽 = 유리\n가로선 = 95% 신뢰구간", c)

    # ③ 남성 내부 분해 — 호칭은 나이·역할의 대리 지표
    cell = gs[0, 2].get_position(fig)
    ys = np.arange(len(tit))[::-1]
    ax3.set_xlim(0, 108)
    ax3.set_ylim(-0.55, len(tit) - 0.45)
    style(ax3, c, grid_axis="x")
    for y, (idx, r) in zip(ys, tit.iterrows()):
        dot_ci(ax3, y, r["rate"], r["lo"], r["hi"], c, color=c["dead"] if idx == "Mr(성인남성)" else c["alive"])
    ax3.set_yticks(ys)
    ax3.set_yticklabels([f"{i}  n={int(n)}" for i, n in zip(tit.index, tit["n"])], fontsize=10, color=c["ink"])
    ax3.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax3, "x")
    master, mr = tit.loc["Master(소년)", "rate"], tit.loc["Mr(성인남성)", "rate"]
    cap(fig, cell, "③ 남성을 쪼개면 — 호칭별 생존율", c)
    note(fig, cell, f"같은 남성이라도 소년 {master:.1f}% vs 성인 {mr:.1f}%\n'여성과 아이 먼저'에서 아이 쪽의 흔적", c)

    if table:
        print("\nH1")
        show("호칭군별 생존율", tit)
        print(f"\n  Cramér's V: 성별 {sex['v']:.3f} · 등급 {cls['v']:.3f}")
        print(f"  오즈비: 여성 {np.exp(res.params['여성']):.2f} · 2등급 {np.exp(res.params['p2']):.2f} · "
              f"3등급 {np.exp(res.params['p3']):.2f}")
    return fig


# ── H2 ───────────────────────────────────────────────────────────────────────
def fig_h2(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """교호작용은 '두 선의 기울기 차이' — 선그래프가 가설 진술 그 자체."""
    base = d[["여성", "p2", "p3"]]
    inter = base.assign(**{"f2": d["여성"] * d["p2"], "f3": d["여성"] * d["p3"]})
    t = lr_test(logit(d["Survived"], inter), logit(d["Survived"], base))

    left = 0.075
    fig = frame(
        c, 13.0, left, "H2", "등급 효과는 성별에 따라 다르게 작동한다",
        "성별과 등급을 따로 더하는 모형으로는 부족하다 — 두 선이 평행한지 보면 된다",
        "지지", True,
        f"우도비 검정(교호작용 항 2개 추가): LR={t['lr']:.1f}, dof={t['dof']}, p={t['p']:.2e}\n"
        "데이터: titanic.csv 891명 · 세로축은 각 칸의 생존율(%)",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.28)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    xs = np.array([0, 1, 2])
    ax1.set_xlim(-0.5, 2.85)
    ax1.set_ylim(0, 108)
    style(ax1, c, grid_axis="y")
    spread = {}
    for sex, color in (("여성", c["alive"]), ("남성", c["series2"])):
        sub = d[d["성별"] == sex]
        rs = [sub[sub["Pclass"] == p]["Survived"].mean() * 100 for p in (1, 2, 3)]
        spread[sex] = rs[0] - rs[2]
        ax1.plot(xs, rs, color=color, linewidth=2, zorder=3, label=sex)
        ax1.plot(xs, rs, "o", markersize=9, color=color, markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax1.text(xs[-1] + 0.12, rs[-1], f"{sex} {rs[-1]:.1f}%", va="center", ha="left",
                 color=color, fontsize=11, fontweight="bold")
        ax1.text(xs[0] - 0.12, rs[0], f"{rs[0]:.1f}%", va="center", ha="right", color=c["ink2"], fontsize=10)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["1등급", "2등급", "3등급"], fontsize=11, color=c["ink"])
    ax1.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax1, "y")
    leg = ax1.legend(frameon=False, fontsize=10, loc="lower left", handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    ax1.annotate("여성은 1·2등급이 거의 같고\n3등급에서만 무너지는 '계단형'", xy=(1.05, 91), xytext=(0.12, 64),
                 color=c["ink2"], fontsize=9.5, linespacing=1.7,
                 arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9})
    ax1.annotate("남성은 1등급만 높고\n2·3등급이 나란히 바닥", xy=(1.55, 14.5), xytext=(0.8, 33),
                 color=c["ink2"], fontsize=9.5, linespacing=1.7,
                 arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9})
    cell = gs[0, 0].get_position(fig)
    cap(fig, cell, "① 성별 안에서 본 등급별 생존율 — 평행하지 않으면 교호작용", c)
    note(fig, cell, "같은 등급 변화(1→3등급)에 두 성별이 다르게 반응한다", c)

    # 기울기 차이를 숫자 하나로: 1등급 − 3등급 낙폭
    cell = gs[0, 1].get_position(fig)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(0, 60)
    style(ax2, c, grid_axis="y")
    rx, ry = _px_to_data(ax2, 4)
    for x, (sex, color) in enumerate((("여성", c["alive"]), ("남성", c["series2"]))):
        v = spread[sex]
        rounded_bar(ax2, x - 0.28, 0, x + 0.28, v, rx, ry, ("tl", "tr"), color)
        ax2.text(x, v + 1.6, f"{v:.1f}%p", ha="center", va="bottom", color=c["ink"], fontsize=12, fontweight="bold")
    ax2.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["여성", "남성"], fontsize=11, color=c["ink"])
    ax2.tick_params(axis="x", labelcolor=c["ink"])
    ax2.set_yticks([0, 20, 40, 60])
    ax2.set_yticklabels(["0", "20", "40", "60%p"])
    cap(fig, cell, "② 같은 낙폭을 숫자로 — 1등급 빼기 3등급", c)
    note(fig, cell, f"등급은 여성에게 {spread['여성'] / spread['남성']:.1f}배 더 크게 작동한다", c)

    if table:
        print("\nH2")
        show("성별×등급 생존율", rates(d, ["성별", "등급"]))
        print(f"\n  낙폭(1등급-3등급): 여성 {spread['여성']:.1f}%p · 남성 {spread['남성']:.1f}%p")
    return fig


# ── H3 ───────────────────────────────────────────────────────────────────────
def fig_h3(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """'판정 불가'도 결과다 — 먼저 검증 가능한 범위를 그리고, 그 안의 결과를 보여준다."""
    cov = d.groupby("등급", observed=True)["선실기록"].agg(n="size", k="sum")
    cov["rate"] = cov["k"] / cov["n"] * 100

    known = d[d["선실기록"] & (d["Pclass"] == 1)]
    deck = rates(known, "갑판")
    small = deck[deck["n"] < 5]
    deck = deck[deck["n"] >= 5]
    chi = chi2_v(known[known["갑판"].isin(deck.index)], "갑판")

    left = 0.075
    fig = frame(
        c, 13.0, left, "H3", "3등급 여성의 급락은 '선내 위치(갑판)'로 설명된다",
        "검정에 들어가기 전에 물어야 할 것 — 그 위치 정보가 애초에 관측되어 있는가",
        "검증 불가", False,
        f"1등급 내부 갑판 비교(n={chi['n']}): χ²={chi['chi2']:.1f}, dof={chi['dof']}, p={chi['p']:.3f}, "
        f"V={chi['v']:.3f} · 표본 5명 미만 갑판 {len(small)}개 제외\n"
        "'기각'이 아니라 '검증 불가'로 적는 이유 — 가설이 틀렸다는 증거가 아니라, 이 데이터로는 물어볼 수 없다는 뜻",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.26)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    xs = np.arange(len(cov))
    ax1.set_xlim(-0.6, len(cov) - 0.4)
    ax1.set_ylim(0, 108)
    style(ax1, c, grid_axis="y")
    rx, ry = _px_to_data(ax1, 4)
    for x, (idx, row) in zip(xs, cov.iterrows()):
        # 관측이 되는 1등급만 강조, 정작 검증 대상인 2·3등급은 비강조
        rounded_bar(ax1, x - 0.3, 0, x + 0.3, row["rate"], rx, ry, ("tl", "tr"),
                    c["alive"] if idx == "1등급" else c["dead"])
        ax1.text(x, row["rate"] + 3, f"{row['rate']:.1f}%", ha="center", va="bottom",
                 color=c["ink"], fontsize=11.5, fontweight="bold")
        ax1.text(x, row["rate"] + 10.5, f"{int(row['k'])}/{int(row['n'])}명", ha="center", va="bottom",
                 color=c["muted"], fontsize=9.5)
    ax1.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(list(cov.index), fontsize=11, color=c["ink"])
    ax1.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax1, "y")
    ax1.annotate("검증 대상인 3등급의 위치 정보가\n사실상 없다 → 판정 자체가 불가능", xy=(1.85, 9), xytext=(0.42, 40),
                 color=c["ink"], fontsize=10, linespacing=1.7,
                 arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9})
    cell = gs[0, 0].get_position(fig)
    cap(fig, cell, "① 검증 가능 범위 — 등급별 선실(Cabin) 기재율", c)
    note(fig, cell, "선실 기록이 있어야 '선내 위치'를 물어볼 수 있다", c)

    ys = np.arange(len(deck))[::-1]
    ax2.set_xlim(0, 112)
    ax2.set_ylim(-0.55, len(deck) - 0.45)
    style(ax2, c, grid_axis="x")
    for y, (idx, r) in zip(ys, deck.iterrows()):
        dot_ci(ax2, y, r["rate"], r["lo"], r["hi"], c)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{i}갑판  n={int(n)}" for i, n in zip(deck.index, deck["n"])], fontsize=10.5, color=c["ink"])
    ax2.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax2, "x")
    cell = gs[0, 1].get_position(fig)
    cap(fig, cell, "② 관측되는 범위 안에서 — 1등급 갑판별 생존율", c)
    note(fig, cell,
         f"1등급 안에서는 갑판 간 차이가 유의하지 않다 (p={chi['p']:.2f})\n"
         "신뢰구간이 서로 크게 겹친다 — 위치로 생존이 갈린 신호는 보이지 않음", c)

    if table:
        print("\nH3")
        print("\n  [등급별 선실 기재율]")
        for idx, row in cov.iterrows():
            print(f"  {idx:<8}{int(row['k']):>5}/{int(row['n']):<5}{row['rate']:>7.1f}%")
        show("1등급 갑판별 생존율", deck)
    return fig


# ── H4 ───────────────────────────────────────────────────────────────────────
def fig_h4(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """역U자는 '연속된 가족수'로 그려야 형태가 보인다. 오른쪽은 교란 통제."""
    g = rates(d.assign(k=d["가족수"].clip(upper=7)), "k")
    labels = [f"{int(i)}인" if i < 7 else "7인+" for i in g.index]

    base = d[["여성", "p2", "p3"]]
    fam = pd.get_dummies(d["가족규모"], prefix="fam", drop_first=True).astype(int)
    t = lr_test(logit(d["Survived"], pd.concat([base, fam], axis=1)), logit(d["Survived"], base))
    w3 = rates(d[(d["성별"] == "여성") & (d["Pclass"] == 3)], "가족규모")

    left = 0.075
    fig = frame(
        c, 13.2, left, "H4", "가족 규모가 생존을 좌우한다 — 2~4인이 가장 유리한 역U자",
        "혼자면 도와줄 사람이 없고, 너무 많으면 서로를 기다리다 놓친다 — 형태부터 확인하고 교란을 걷어낸다",
        "지지", True,
        f"성별·등급 통제 후 우도비 검정: LR={t['lr']:.1f}, dof={t['dof']}, p={t['p']:.2e}\n"
        "세로선·가로선 = Wilson 95% 신뢰구간 · 7인 이상은 표본이 작아 한 칸으로 묶음",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.3)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    xs = np.arange(len(g))
    ax1.set_xlim(-0.65, len(g) - 0.35)
    ax1.set_ylim(0, 108)
    style(ax1, c, grid_axis="y")
    peak = int(np.argmax(g["rate"].to_numpy()))
    for x, (idx, r) in zip(xs, g.iterrows()):
        # 이야기의 주인공(정점)만 파랑
        vbar_ci(ax1, x, r["rate"], r["lo"], r["hi"], c, c["alive"] if x == peak else c["dead"])
    ax1.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{lab}\nn={int(n)}" for lab, n in zip(labels, g["n"])],
                        fontsize=10, color=c["ink"], linespacing=1.7)
    ax1.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax1, "y")
    ax1.annotate("정점", xy=(peak, g["hi"].iloc[peak] + 1), xytext=(peak, g["hi"].iloc[peak] + 17),
                 color=c["ink2"], fontsize=10, ha="center",
                 arrowprops={"arrowstyle": "-", "color": c["axis"], "linewidth": 0.9})
    cell = gs[0, 0].get_position(fig)
    cap(fig, cell, "① 형태 확인 — 가족 수별 생존율", c)
    note(fig, cell, "가로축 = 본인 포함 동승 가족 수(SibSp + Parch + 1) · 혼자와 대가족이 양쪽 끝에서 함께 낮다",
         c, dy=0.092)

    ys = np.arange(len(w3))[::-1]
    ax2.set_xlim(0, 112)
    ax2.set_ylim(-0.55, len(w3) - 0.45)
    style(ax2, c, grid_axis="x")
    for y, (idx, r) in zip(ys, w3.iterrows()):
        dot_ci(ax2, y, r["rate"], r["lo"], r["hi"], c)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{i}\nn={int(n)}" for i, n in zip(w3.index, w3["n"])],
                        fontsize=10, color=c["ink"], linespacing=1.7)
    ax2.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax2, "x")
    cell = gs[0, 1].get_position(fig)
    cap(fig, cell, "② 교란 통제 — 3등급 여성 안에서만", c)
    note(fig, cell, "성별·등급이 같은 사람들끼리 비교해도 대가족의 급락은 남는다\n"
                    "→ '가족 규모'가 등급의 다른 이름인 것은 아니다", c)

    if table:
        print("\nH4")
        show("가족 수별 생존율", g.set_axis(labels))
        show("3등급 여성 · 가족규모별", w3)
    return fig


# ── H5 ───────────────────────────────────────────────────────────────────────
def fig_h5(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """대리변수 의심이 먼저다 — 등급 구성을 보여준 뒤, 등급을 고정해도 남는지 본다."""
    e = d.dropna(subset=["Embarked"])
    base = e[["여성", "p2", "p3"]]
    emb = pd.get_dummies(e["Embarked"], prefix="emb", drop_first=True).astype(int)
    t = lr_test(logit(e["Survived"], pd.concat([base, emb], axis=1)), logit(e["Survived"], base))

    names = {"C": "C 셰르부르", "Q": "Q 퀸스타운", "S": "S 사우샘프턴"}
    order = ["C", "Q", "S"]
    mix = pd.crosstab(e["Embarked"], e["등급"], normalize="index") * 100
    n_emb = e["Embarked"].value_counts()
    w3 = rates(e[(e["성별"] == "여성") & (e["Pclass"] == 3)], "Embarked", order=order)

    left = 0.115
    fig = frame(
        c, 13.6, left, "H5", "승선항 효과는 등급의 대리변수가 아니다",
        "셰르부르 승객의 생존율이 높은 건 1등급이 많아서일 수 있다 — 그렇다면 등급을 고정하면 사라져야 한다",
        "지지", True,
        f"성별·등급 통제 후 우도비 검정: LR={t['lr']:.1f}, dof={t['dof']}, p={t['p']:.3f} "
        f"(승선항 결측 {int(d['Embarked'].isna().sum())}명 제외) · 유의한 대비는 S vs C\n"
        "오른쪽은 표본이 작아 신뢰구간이 넓다 — 순위만 읽지 말고 구간의 겹침을 함께 볼 것",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.32)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ① 등급 구성: 순서형 한 색 램프(진할수록 상위 등급)
    ys = np.arange(len(order))[::-1]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.75, len(order) - 0.25)
    style(ax1, c, grid_axis="x")
    rx, ry = _px_to_data(ax1, 4)
    gap, _ = _px_to_data(ax1, 2)  # 조각 사이 2px 서피스 간격
    for y, port in zip(ys, order):
        x0 = 0.0
        for j, cls in enumerate(["1등급", "2등급", "3등급"]):
            x1 = x0 + mix.loc[port, cls]
            corners = ("bl", "tl") if j == 0 else (("br", "tr") if j == 2 else ())
            rounded_bar(ax1, x0 + (gap / 2 if j else 0), y - 0.3, max(x1 - gap / 2, x0 + gap), y + 0.3,
                        rx, ry, corners, c["ramp"][j])
            if x1 - x0 >= 11:
                # 글자색은 모드가 아니라 '조각의 밝기'를 따라간다 — 가장 진한 단계에만 흰 글자
                ax1.text((x0 + x1) / 2, y, f"{x1 - x0:.0f}%", ha="center", va="center",
                         color="#ffffff" if j == 0 else "#0b0b0b", fontsize=10.5, fontweight="bold", zorder=5)
            x0 = x1
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{names[p]}\nn={int(n_emb[p])}" for p in order],
                        fontsize=10.5, color=c["ink"], linespacing=1.7)
    ax1.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax1, "x")
    cell = gs[0, 0].get_position(fig)
    # 범례는 캡션 줄 오른쪽에 — 아래에 두면 주석·꼬리말과 부딪힌다
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=c["ramp"][j], label=lab)
               for j, lab in enumerate(["1등급", "2등급", "3등급"])]
    leg = fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(cell.x1, cell.y1 + 0.028),
                     frameon=False, ncol=3, fontsize=9.5, handletextpad=0.3, columnspacing=1.5)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    cap(fig, cell, "① 의심의 근거 — 승선항별 등급 구성", c)
    note(fig, cell, f"셰르부르는 1등급이 {mix.loc['C', '1등급']:.0f}%, 퀸스타운은 3등급이 {mix.loc['Q', '3등급']:.0f}%\n"
                    "→ 승선항 차이가 등급 차이의 다른 이름일 수 있다", c)

    ys2 = np.arange(len(w3))[::-1]
    ax2.set_xlim(0, 112)
    ax2.set_ylim(-0.55, len(w3) - 0.45)
    style(ax2, c, grid_axis="x")
    for y, (idx, r) in zip(ys2, w3.iterrows()):
        dot_ci(ax2, y, r["rate"], r["lo"], r["hi"], c)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([f"{names[i]}\nn={int(n)}" for i, n in zip(w3.index, w3["n"])],
                        fontsize=10, color=c["ink"], linespacing=1.7)
    ax2.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax2, "x")
    cell = gs[0, 1].get_position(fig)
    cap(fig, cell, "② 등급을 고정한 뒤 — 3등급 여성 생존율", c)
    note(fig, cell, "'3등급 여성' 하나로 고정해도 격차가 남는다\n"
                    "→ 승선항에는 등급으로 환원되지 않는 무언가가 실려 있다", c)

    if table:
        print("\nH5")
        print("\n  [승선항별 등급 구성 %]")
        print(mix.round(1).to_string())
        show("3등급 여성 · 승선항별", w3)
    return fig


# ── H6 ───────────────────────────────────────────────────────────────────────
def fig_h6(d: pd.DataFrame, c: dict, table: bool) -> plt.Figure:
    """결측 '여부'가 결과와 얼마나 벌어져 있는지(덤벨) + 그 격차가 어디서 왔는지(출처)."""
    base = d[["여성", "p2", "p3"]]
    specs = []
    for col, label in (("선실기록", "선실(Cabin)"), ("나이결측", "나이(Age)")):
        t = lr_test(logit(d["Survived"], base.assign(**{col: d[col].astype(int)})), logit(d["Survived"], base))
        flag = d[col] if col == "선실기록" else ~d[col]  # 둘 다 '기재됨=True'로 맞춘다
        specs.append((label, d[flag]["Survived"].mean() * 100, d[~flag]["Survived"].mean() * 100, t))

    left = 0.105
    fig = frame(
        c, 13.4, left, "H6", "결측 자체가 생존과 상관있다 — 다만 두 결측의 성격은 다르다",
        "빈칸은 '정보 없음'이 아니라 그 자체로 정보다 — 격차를 보고, 그 격차가 등급으로 설명되는지 따진다",
        "지지", True,
        "성별·등급을 통제하면 갈린다 — 나이 결측은 효과가 사라져 MAR에 가깝고, 선실 결측은 남아 MNAR이 의심된다\n"
        "실무 함의: 나이는 등급·호칭별 조건부 대치가 안전하고, 선실 결측 여부는 누출(leakage)성 피처일 수 있어 해석용 모델에서는 배제",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=left, right=R,
                          top=TOP, bottom=BOTTOM, wspace=0.3)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    ys = np.arange(len(specs))[::-1]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.7, len(specs) - 0.3)
    style(ax1, c, grid_axis="x")
    for y, (label, ry_, rn, t) in zip(ys, specs):
        ax1.plot([rn, ry_], [y, y], color=c["axis"], linewidth=2, solid_capstyle="round", zorder=2)
        ax1.plot([rn], [y], "o", markersize=9, color=c["dead"],
                 markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax1.plot([ry_], [y], "o", markersize=9, color=c["alive"],
                 markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        ax1.text(min(rn, ry_) - 2.5, y, f"{min(rn, ry_):.1f}%", va="center", ha="right", color=c["ink2"], fontsize=10.5)
        ax1.text(max(rn, ry_) + 2.5, y, f"{max(rn, ry_):.1f}%", va="center", ha="left",
                 color=c["ink"], fontsize=11, fontweight="bold")
        tail = "통제 후 효과 소멸 → MAR" if t["p"] >= 0.05 else "통제 후에도 유의 → MNAR 의심"
        ax1.text(min(rn, ry_), y - 0.22, f"성별·등급 통제 후 p={t['p']:.3f} · {tail}",
                 va="top", ha="left", color=c["muted"], fontsize=9.5)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([s[0] for s in specs], fontsize=11, color=c["ink"])
    ax1.tick_params(axis="y", labelcolor=c["ink"])
    pct_axis(ax1, "x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=c["alive"], label="기재됨"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=c["dead"], label="미기재")]
    leg = ax1.legend(handles=handles, frameon=False, fontsize=10, loc="upper right", ncol=2, handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    cell = gs[0, 0].get_position(fig)
    cap(fig, cell, "① 격차 — 기재 여부에 따른 생존율", c)
    note(fig, cell, "두 결측 모두 단변량으로는 생존과 연관된다(MCAR 기각)", c)

    # 격차의 출처: 결측이 등급에 몰려 있으면 등급 통제로 사라진다
    miss = pd.DataFrame({
        "나이": d.groupby("등급", observed=True)["나이결측"].mean() * 100,
        "선실": (1 - d.groupby("등급", observed=True)["선실기록"].mean()) * 100,
    })
    xs = np.arange(len(miss))
    ax2.set_xlim(-0.6, len(miss) - 0.4)
    ax2.set_ylim(0, 112)
    style(ax2, c, grid_axis="y")
    rx, ry = _px_to_data(ax2, 4)
    gap, _ = _px_to_data(ax2, 2)
    for x, (idx, row) in zip(xs, miss.iterrows()):
        for j, (key, color) in enumerate((("나이", c["alive"]), ("선실", c["series2"]))):
            x0 = x - 0.32 + j * 0.32 + (gap / 2 if j else 0)
            rounded_bar(ax2, x0, 0, x0 + 0.32 - gap / 2, row[key], rx, ry, ("tl", "tr"), color)
            ax2.text(x0 + 0.16, row[key] + 2.5, f"{row[key]:.0f}%", ha="center", va="bottom",
                     color=c["ink"], fontsize=10, fontweight="bold")
    ax2.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(list(miss.index), fontsize=11, color=c["ink"])
    ax2.tick_params(axis="x", labelcolor=c["ink"])
    pct_axis(ax2, "y")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=col, label=f"{lab} 미기재율")
               for lab, col in (("나이", c["alive"]), ("선실", c["series2"]))]
    leg = ax2.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper left", handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    cell = gs[0, 1].get_position(fig)
    cap(fig, cell, "② 격차의 출처 — 등급별 미기재율", c)
    note(fig, cell, "나이 결측은 3등급에 몰려 있어 등급을 알면 설명된다\n"
                    "선실 결측은 2·3등급 모두 90%를 넘어 등급만으로는 설명되지 않는다", c)

    if table:
        print("\nH6")
        print("\n  [기재 여부별 생존율]")
        for label, ry_, rn, t in specs:
            print(f"  {label:<12} 기재됨 {ry_:>5.1f}% · 미기재 {rn:>5.1f}% · 통제 후 p={t['p']:.4f}")
        print("\n  [등급별 미기재율 %]")
        print(miss.round(1).to_string())
    return fig


# ── 실행 ─────────────────────────────────────────────────────────────────────
BUILDERS = {"H1": fig_h1, "H2": fig_h2, "H3": fig_h3, "H4": fig_h4, "H5": fig_h5, "H6": fig_h6}


def render(d: pd.DataFrame, tags: list[str], dark: bool, table: bool, outdir: Path) -> None:
    c = dict(DARK if dark else LIGHT, **(EXTRA_DARK if dark else EXTRA_LIGHT))
    for tag in tags:
        fig = BUILDERS[tag](d, c, table)
        out = outdir / (f"{tag}_dark.png" if dark else f"{tag}.png")
        fig.savefig(out, facecolor=c["surface"])
        plt.close(fig)
        print(f"저장 완료: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="가설별 전용 시각화 (H1~H6, 가설당 한 장)")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--both", action="store_true", help="라이트·다크 둘 다 렌더링")
    parser.add_argument("--only", nargs="+", metavar="TAG", help="특정 가설만 (예: --only H3 H5)")
    parser.add_argument("--table", action="store_true", help="그림의 수치를 콘솔 표로도 출력")
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="저장 폴더")
    args = parser.parse_args()

    tags = [t.upper() for t in (args.only or BUILDERS)]
    unknown = [t for t in tags if t not in BUILDERS]
    if unknown:
        parser.error(f"알 수 없는 가설: {', '.join(unknown)} (가능: {', '.join(BUILDERS)})")

    setup_font()
    args.outdir.mkdir(parents=True, exist_ok=True)
    d = prepare(load_data())

    modes = [False, True] if args.both else [args.dark]
    for k, dark in enumerate(modes):
        render(d, tags, dark, args.table and k == 0, args.outdir)


if __name__ == "__main__":
    main()
