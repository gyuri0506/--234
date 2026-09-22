# -*- coding: utf-8 -*-
"""성별 생존 비율을 파이(도넛) 차트로 비교한다.

여성/남성 각각의 '해당 성별 내 생존/사망 구성'을 같은 크기의 파이 두 개로 나란히 놓고,
가운데에 생존율을 숫자로 넣어 두 파이의 차이를 숫자로도 읽을 수 있게 했다.
배색·폰트·레이블 규칙은 titanic_survival.py 와 동일하다.

사용법:
    python titanic_sex_pie.py           # 라이트 모드 -> titanic_sex_pie.png
    python titanic_sex_pie.py --dark    # 다크 모드   -> titanic_sex_pie_dark.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from titanic_survival import (
    ALIVE_LABEL,
    DARK,
    DEAD_LABEL,
    LIGHT,
    load_data,
    panel_title,
    rate_table,
    setup_font,
)

HERE = Path(__file__).resolve().parent


def draw_pie(
    ax: plt.Axes,
    cell,
    label: str,
    total: int,
    alive: int,
    c: dict,
    hero_size: float = 30,
    label_size: float = 10.5,
) -> None:
    dead = total - alive
    rate = alive / total * 100

    ax.set_facecolor(c["surface"])
    ax.set_aspect("equal")
    wedges, _ = ax.pie(
        [alive, dead],
        startangle=90,
        counterclock=False,
        colors=[c["alive"], c["dead"]],
        wedgeprops={"width": 0.36, "edgecolor": c["surface"], "linewidth": 2},
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)

    # 가운데 생존율: 두 파이를 눈대중으로 비교하지 않아도 되게 숫자로 박아둔다
    ax.text(0, 0.1, f"{rate:.1f}%", ha="center", va="center", color=c["ink"], fontsize=hero_size, fontweight="bold")
    ax.text(0, -0.25, "생존율", ha="center", va="center", color=c["ink2"], fontsize=label_size * 0.95)

    for wedge, name, n in ((wedges[0], ALIVE_LABEL, alive), (wedges[1], DEAD_LABEL, dead)):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ax.text(
            x * 1.17,
            y * 1.17,
            f"{name} {n:,}명\n{n / total * 100:.1f}%",
            ha="left" if x >= 0 else "right",
            va="center",
            color=c["ink"] if name == ALIVE_LABEL else c["ink2"],
            fontsize=label_size,
            fontweight="bold" if name == ALIVE_LABEL else "normal",
            linespacing=1.5,
        )

    panel_title(ax.figure, cell, f"{label}", f"탑승 {total:,}명 · 생존 {alive:,}명", c)


def build_figure(dark: bool) -> tuple[plt.Figure, "object"]:
    c = DARK if dark else LIGHT
    df = load_data()
    tbl = rate_table(df, "성별", ["여성", "남성"])

    fig = plt.figure(figsize=(11.6, 6.6), dpi=140, facecolor=c["surface"])
    gs = fig.add_gridspec(1, 2, left=0.055, right=0.955, top=0.70, bottom=0.10, wspace=0.10)

    fig.text(0.055, 0.945, "성별 생존 비율", color=c["ink"], fontsize=21, fontweight="bold", va="top")
    fig.text(
        0.055,
        0.892,
        "각 성별 안에서의 생존/사망 구성 · 탑승객 891명 (성별 기재 누락 없음)",
        color=c["ink2"],
        fontsize=11.5,
        va="top",
    )

    legend = fig.legend(
        handles=[Patch(facecolor=c["alive"], label=ALIVE_LABEL), Patch(facecolor=c["dead"], label=DEAD_LABEL)],
        loc="upper right",
        bbox_to_anchor=(0.955, 0.955),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
        fontsize=11,
    )
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    for k, (name, row) in enumerate(tbl.iterrows()):
        ax = fig.add_subplot(gs[0, k])
        draw_pie(ax, gs[0, k].get_position(fig), str(name), int(row["전체"]), int(row["생존"]), c)

    ratio = tbl.loc["여성", "생존율"] / tbl.loc["남성", "생존율"]
    fig.text(
        0.055,
        0.045,
        f"여성 {tbl.loc['여성', '생존율']:.1f}%  vs  남성 {tbl.loc['남성', '생존율']:.1f}%  ·  여성 생존율이 약 {ratio:.1f}배",
        color=c["ink"],
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.012,
        "데이터: titanic.csv · 각 파이는 해당 성별 내 비율이며, 두 파이의 크기는 인원수를 뜻하지 않음",
        color=c["muted"],
        fontsize=9,
    )
    return fig, tbl


def main() -> None:
    parser = argparse.ArgumentParser(description="성별 생존 비율 파이 차트")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로")
    args = parser.parse_args()

    setup_font()
    c = DARK if args.dark else LIGHT
    fig, tbl = build_figure(dark=args.dark)
    out = args.out or HERE / ("titanic_sex_pie_dark.png" if args.dark else "titanic_sex_pie.png")
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)

    print(f"저장 완료: {out}")
    print(f"\n[표 보기] 성별 생존 비율")
    print(f"{'성별':<6} {'전체':>6} {'생존':>6} {'사망':>6} {'생존율':>8}")
    print("-" * 38)
    for name, row in tbl.iterrows():
        print(f"{str(name):<6} {int(row['전체']):>6} {int(row['생존']):>6} {int(row['사망']):>6} {row['생존율']:>7.1f}%")


if __name__ == "__main__":
    main()
