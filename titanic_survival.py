# -*- coding: utf-8 -*-
"""타이타닉 탑승객 생존 여부 시각화.

titanic.csv 를 읽어 생존 여부를 중심으로 4개 패널을 한 장으로 구성한다.
  1) 전체 생존 비율 도넛(파이) + 생존율 히어로 숫자
  2) 연령대별 생존율
  3) 성별 생존/사망 구성비 (100% 누적 막대)
  4) 객실 등급별 생존/사망 구성비 (100% 누적 막대)

사용법:
    python titanic_survival.py              # 라이트 모드 -> titanic_survival.png
    python titanic_survival.py --dark       # 다크 모드   -> titanic_survival_dark.png
    python titanic_survival.py --out x.png  # 저장 경로 지정
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "titanic.csv"

# ── 배색 ─────────────────────────────────────────────────────────────────────
# 강조(emphasis) 배색: 이야기의 주인공인 '생존'만 파랑, '사망'은 비강조 그레이.
# 두 값 모두 검증된 기준 팔레트의 값이며, 색상 하나(파랑) + 중립 그레이 조합이라
# 색약 환경에서도 명도 차이로 구분된다. 모든 조각에 직접 레이블을 붙여
# 색에만 의존하지 않도록 했다.
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "alive": "#2a78d6",
    "dead": "#c3c2b7",
    "on_alive": "#ffffff",
    "on_dead": "#52514e",
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "alive": "#3987e5",
    "dead": "#898781",
    "on_alive": "#ffffff",
    "on_dead": "#1a1a19",
}

ALIVE_LABEL = "생존"
DEAD_LABEL = "사망"

AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]
AGE_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


# ── 한글 폰트 ────────────────────────────────────────────────────────────────
def setup_font() -> None:
    for family in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams["axes.unicode_minus"] = False


# ── 모서리가 둥근 막대 ───────────────────────────────────────────────────────
def _px_to_data(ax: plt.Axes, px: float) -> tuple[float, float]:
    """픽셀 길이를 해당 축의 x/y 데이터 단위로 환산한다."""
    fig = ax.figure
    pos = ax.get_position()
    w_px = pos.width * fig.get_figwidth() * fig.dpi
    h_px = pos.height * fig.get_figheight() * fig.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return px * (x1 - x0) / w_px, px * (y1 - y0) / h_px


def rounded_bar(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    corners: tuple[str, ...],
    color: str,
    zorder: float = 3,
) -> None:
    """데이터 끝단만 둥글게 처리한 막대(corners: bl/br/tr/tl)를 그린다."""
    k = 0.5523  # 원에 가까운 베지어 제어점 계수
    rx = max(min(rx, (x1 - x0) / 2), 0.0)
    ry = max(min(ry, (y1 - y0) / 2), 0.0)

    def r(name: str) -> tuple[float, float]:
        return (rx, ry) if name in corners else (0.0, 0.0)

    bl, br, tr, tl = r("bl"), r("br"), r("tr"), r("tl")
    verts: list[tuple[float, float]] = [(x0 + bl[0], y0)]
    codes: list[int] = [MplPath.MOVETO]

    def line(pt: tuple[float, float]) -> None:
        verts.append(pt)
        codes.append(MplPath.LINETO)

    def arc(c1: tuple[float, float], c2: tuple[float, float], end: tuple[float, float]) -> None:
        verts.extend([c1, c2, end])
        codes.extend([MplPath.CURVE4] * 3)

    line((x1 - br[0], y0))
    if br[0] or br[1]:
        arc((x1 - br[0] * (1 - k), y0), (x1, y0 + br[1] * (1 - k)), (x1, y0 + br[1]))
    line((x1, y1 - tr[1]))
    if tr[0] or tr[1]:
        arc((x1, y1 - tr[1] * (1 - k)), (x1 - tr[0] * (1 - k), y1), (x1 - tr[0], y1))
    line((x0 + tl[0], y1))
    if tl[0] or tl[1]:
        arc((x0 + tl[0] * (1 - k), y1), (x0, y1 - tl[1] * (1 - k)), (x0, y1 - tl[1]))
    line((x0, y0 + bl[1]))
    if bl[0] or bl[1]:
        arc((x0, y0 + bl[1] * (1 - k)), (x0 + bl[0] * (1 - k), y0), (x0 + bl[0], y0))
    verts.append((x0 + bl[0], y0))
    codes.append(MplPath.CLOSEPOLY)

    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0, zorder=zorder))


# ── 데이터 ───────────────────────────────────────────────────────────────────
def load_data(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["성별"] = df["Sex"].map({"female": "여성", "male": "남성"})
    df["등급"] = df["Pclass"].map({1: "1등급", 2: "2등급", 3: "3등급"})
    df["연령대"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def rate_table(df: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    """그룹별 인원 수·생존 수·생존율 표."""
    g = df.groupby(by, observed=True)["Survived"].agg(전체="size", 생존="sum")
    g = g.reindex([o for o in order if o in g.index])
    g["사망"] = g["전체"] - g["생존"]
    g["생존율"] = g["생존"] / g["전체"] * 100
    return g


# ── 공통 축 스타일 ───────────────────────────────────────────────────────────
def style_axes(ax: plt.Axes, c: dict, grid_axis: str | None = None) -> None:
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=c["grid"], linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=c["muted"], length=0, labelsize=9.5)


def panel_title(fig: plt.Figure, cell, title: str, subtitle: str, c: dict) -> None:
    """그리드 칸 기준으로 제목/부제를 찍는다(축 박스가 정사각으로 줄어도 정렬 유지)."""
    fig.text(cell.x0, cell.y1 + 0.043, title, color=c["ink"], fontsize=12.5, fontweight="bold", va="bottom")
    fig.text(cell.x0, cell.y1 + 0.014, subtitle, color=c["ink2"], fontsize=9.5, va="bottom")


# ── 패널 1: 전체 생존 비율 도넛 ──────────────────────────────────────────────
def panel_donut(ax: plt.Axes, cell, df: pd.DataFrame, c: dict) -> None:
    total = len(df)
    alive = int(df["Survived"].sum())
    dead = total - alive
    rate = alive / total * 100

    ax.set_facecolor(c["surface"])
    ax.set_aspect("equal")
    wedges, _ = ax.pie(
        [alive, dead],
        startangle=90,
        counterclock=False,
        colors=[c["alive"], c["dead"]],
        wedgeprops={"width": 0.34, "edgecolor": c["surface"], "linewidth": 2},
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)

    # 히어로 숫자: 도넛 가운데가 곧 헤드라인
    ax.text(0, 0.1, f"{rate:.1f}%", ha="center", va="center", color=c["ink"], fontsize=33, fontweight="bold")
    ax.text(0, -0.24, "전체 생존율", ha="center", va="center", color=c["ink2"], fontsize=10.5)

    # 조각별 직접 레이블 (색에만 의존하지 않도록)
    for wedge, name, n in ((wedges[0], ALIVE_LABEL, alive), (wedges[1], DEAD_LABEL, dead)):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ax.text(
            x * 1.16,
            y * 1.16,
            f"{name}\n{n:,}명 · {n / total * 100:.1f}%",
            ha="left" if x >= 0 else "right",
            va="center",
            color=c["ink"] if name == ALIVE_LABEL else c["ink2"],
            fontsize=10,
            fontweight="bold" if name == ALIVE_LABEL else "normal",
            linespacing=1.5,
        )
    panel_title(ax.figure, cell, "전체 생존 여부", f"탑승객 {total:,}명 중 {alive:,}명 생존", c)


# ── 패널 2: 연령대별 생존율 ──────────────────────────────────────────────────
def panel_age(ax: plt.Axes, cell, df: pd.DataFrame, c: dict) -> None:
    tbl = rate_table(df.dropna(subset=["연령대"]), "연령대", AGE_LABELS)
    n_missing = int(df["Age"].isna().sum())

    xs = np.arange(len(tbl))
    ax.set_xlim(-0.6, len(tbl) - 0.4)
    ax.set_ylim(0, 100)
    style_axes(ax, c, grid_axis="y")
    rx, ry = _px_to_data(ax, 4)

    half = 0.33
    for x, rate in zip(xs, tbl["생존율"]):
        rounded_bar(ax, x - half, 0, x + half, rate, rx, ry, ("tl", "tr"), c["alive"])
        ax.text(x, rate + 3.5, f"{rate:.0f}%", ha="center", va="bottom", color=c["ink2"], fontsize=9.5)

    ax.axhline(0, color=c["axis"], linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\nn={int(n)}" for i, n in zip(tbl.index, tbl["전체"])], fontsize=9, linespacing=1.6)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    panel_title(ax.figure, cell, "연령대별 생존율", f"나이 미기재 {n_missing}명 제외 · 10세 미만이 가장 높음", c)


# ── 패널 3·4: 100% 누적 막대 ─────────────────────────────────────────────────
def panel_stacked(ax: plt.Axes, cell, tbl: pd.DataFrame, title: str, subtitle: str, c: dict) -> None:
    ys = np.arange(len(tbl))[::-1]  # 표의 첫 행이 맨 위
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(tbl) - 0.3)
    style_axes(ax, c, grid_axis="x")
    rx, ry = _px_to_data(ax, 4)
    gap, _ = _px_to_data(ax, 2)  # 조각 사이 2px 서피스 간격

    half = 0.3 if len(tbl) > 2 else 0.26
    for y, (_, row) in zip(ys, tbl.iterrows()):
        rate = row["생존율"]
        rounded_bar(ax, 0, y - half, max(rate - gap / 2, gap), y + half, rx, ry, ("bl", "tl"), c["alive"])
        rounded_bar(ax, min(rate + gap / 2, 100 - gap), y - half, 100, y + half, rx, ry, ("br", "tr"), c["dead"])

        # 조각 안에 들어갈 폭이면 안에, 아니면 조각 바깥에 레이블
        if rate >= 14:
            ax.text(rate - 2.5, y, f"{rate:.0f}%", ha="right", va="center",
                    color=c["on_alive"], fontsize=10.5, fontweight="bold", zorder=4)
        else:
            ax.text(rate + 2.5, y, f"{rate:.0f}%", ha="left", va="center",
                    color=c["ink"], fontsize=10.5, fontweight="bold", zorder=4)
        if 100 - rate >= 14:
            ax.text(100 - 2.5, y, f"{100 - rate:.0f}%", ha="right", va="center",
                    color=c["on_dead"], fontsize=10, zorder=4)

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{i}\nn={int(n)}" for i, n in zip(tbl.index, tbl["전체"])],
        fontsize=10.5,
        color=c["ink"],
        linespacing=1.6,
    )
    ax.tick_params(axis="y", labelcolor=c["ink"])
    panel_title(ax.figure, cell, title, subtitle, c)


# ── 표 보기 (색 없이도 모든 값을 읽을 수 있는 대체 경로) ─────────────────────
def print_table(df: pd.DataFrame) -> None:
    total, alive = len(df), int(df["Survived"].sum())
    print(f"\n[표 보기] 타이타닉 생존 여부 · 탑승객 {total:,}명 · 생존 {alive:,}명 ({alive / total * 100:.1f}%)")
    blocks = [
        ("성별", rate_table(df, "성별", ["여성", "남성"])),
        ("객실 등급", rate_table(df, "등급", ["1등급", "2등급", "3등급"])),
        ("연령대", rate_table(df.dropna(subset=["연령대"]), "연령대", AGE_LABELS)),
    ]
    for name, tbl in blocks:
        print(f"\n{name:<8} {'전체':>6} {'생존':>6} {'사망':>6} {'생존율':>8}")
        print("-" * 40)
        for idx, row in tbl.iterrows():
            print(f"{str(idx):<8} {int(row['전체']):>6} {int(row['생존']):>6} {int(row['사망']):>6} {row['생존율']:>7.1f}%")


# ── 그림 구성 ────────────────────────────────────────────────────────────────
def build_figure(df: pd.DataFrame, dark: bool) -> plt.Figure:
    c = DARK if dark else LIGHT
    total, alive = len(df), int(df["Survived"].sum())

    fig = plt.figure(figsize=(13.0, 9.4), dpi=140, facecolor=c["surface"])
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.965, top=0.785, bottom=0.075, wspace=0.26, hspace=0.44)

    fig.text(0.065, 0.945, "타이타닉 탑승객 생존 여부", color=c["ink"], fontsize=21, fontweight="bold", va="top")
    fig.text(
        0.065,
        0.897,
        f"탑승객 {total:,}명 중 {alive:,}명 생존 · 성별·객실 등급·연령대별 분해",
        color=c["ink2"],
        fontsize=11.5,
        va="top",
    )

    legend = fig.legend(
        handles=[
            Patch(facecolor=c["alive"], label=ALIVE_LABEL),
            Patch(facecolor=c["dead"], label=DEAD_LABEL),
        ],
        loc="upper right",
        bbox_to_anchor=(0.965, 0.955),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
        fontsize=11,
    )
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    slots = [(i, j) for i in (0, 1) for j in (0, 1)]
    axes = [fig.add_subplot(gs[i, j]) for i, j in slots]
    cells = [gs[i, j].get_position(fig) for i, j in slots]

    panel_donut(axes[0], cells[0], df, c)
    panel_age(axes[1], cells[1], df, c)
    panel_stacked(
        axes[2],
        cells[2],
        rate_table(df, "성별", ["여성", "남성"]),
        "성별 생존 여부",
        "여성 생존율이 남성의 약 4배",
        c,
    )
    panel_stacked(
        axes[3],
        cells[3],
        rate_table(df, "등급", ["1등급", "2등급", "3등급"]),
        "객실 등급별 생존 여부",
        "등급이 낮아질수록 생존율 하락",
        c,
    )

    fig.text(
        0.065,
        0.028,
        "데이터: titanic.csv · 모든 값은 콘솔의 [표 보기]에서 숫자로도 확인 가능",
        color=c["muted"],
        fontsize=9,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="타이타닉 생존 여부 시각화")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로 (기본: titanic_survival[_dark].png)")
    parser.add_argument("--no-table", action="store_true", help="콘솔 표 출력 생략")
    args = parser.parse_args()

    setup_font()
    df = load_data()

    out = args.out or HERE / (f"titanic_survival_dark.png" if args.dark else "titanic_survival.png")
    c = DARK if args.dark else LIGHT
    fig = build_figure(df, dark=args.dark)
    fig.savefig(out, facecolor=c["surface"], bbox_inches=None)
    plt.close(fig)
    print(f"저장 완료: {out}")

    if not args.no_table:
        print_table(df)


if __name__ == "__main__":
    main()
