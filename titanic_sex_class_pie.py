# -*- coding: utf-8 -*-
"""성별 × 객실 등급(Pclass) 생존 비율을 파이(도넛) 차트 6개로 비교한다.

2행(여성/남성) × 3열(1/2/3등급) 격자. 각 파이는 '해당 집단 안에서의 생존/사망 구성'이며,
가운데 숫자는 그 집단의 생존율이다. 같은 열끼리 보면 등급 효과, 같은 행끼리 보면
성별 효과가 읽힌다.

사용법:
    python titanic_sex_class_pie.py           # 라이트 모드 -> titanic_sex_class_pie.png
    python titanic_sex_class_pie.py --dark    # 다크 모드   -> titanic_sex_class_pie_dark.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from titanic_sex_pie import draw_pie
from titanic_survival import ALIVE_LABEL, DARK, DEAD_LABEL, LIGHT, load_data, setup_font

HERE = Path(__file__).resolve().parent

SEXES = ["여성", "남성"]
CLASSES = ["1등급", "2등급", "3등급"]


def grid_table(df):
    """성별 × 등급별 인원 수 / 생존 수 / 생존율."""
    g = df.groupby(["성별", "등급"], observed=True)["Survived"].agg(전체="size", 생존="sum")
    g["사망"] = g["전체"] - g["생존"]
    g["생존율"] = g["생존"] / g["전체"] * 100
    return g


def build_figure(dark: bool):
    c = DARK if dark else LIGHT
    df = load_data()
    tbl = grid_table(df)

    fig = plt.figure(figsize=(14.2, 9.6), dpi=140, facecolor=c["surface"])
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.955, top=0.755, bottom=0.105, wspace=0.14, hspace=0.42)

    fig.text(0.055, 0.955, "성별 × 객실 등급별 생존 비율", color=c["ink"], fontsize=21, fontweight="bold", va="top")
    fig.text(
        0.055,
        0.906,
        "각 집단 안에서의 생존/사망 구성 · 탑승객 891명 · 가운데 숫자는 그 집단의 생존율",
        color=c["ink2"],
        fontsize=11.5,
        va="top",
    )

    legend = fig.legend(
        handles=[Patch(facecolor=c["alive"], label=ALIVE_LABEL), Patch(facecolor=c["dead"], label=DEAD_LABEL)],
        loc="upper right",
        bbox_to_anchor=(0.955, 0.962),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
        fontsize=11,
    )
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    for row, sex in enumerate(SEXES):
        for col, cls in enumerate(CLASSES):
            r = tbl.loc[(sex, cls)]
            ax = fig.add_subplot(gs[row, col])
            draw_pie(
                ax,
                gs[row, col].get_position(fig),
                f"{sex} · {cls}",
                int(r["전체"]),
                int(r["생존"]),
                c,
                hero_size=24,
                label_size=9.5,
            )

    lo = tbl["생존율"].idxmin()
    hi = tbl["생존율"].idxmax()
    fig.text(
        0.055,
        0.052,
        f"최고 {hi[0]}·{hi[1]} {tbl.loc[hi, '생존율']:.1f}%   ↔   최저 {lo[0]}·{lo[1]} {tbl.loc[lo, '생존율']:.1f}%"
        f"   ·   같은 등급 안에서도 성별 격차가 등급 격차보다 큼",
        color=c["ink"],
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.018,
        "데이터: titanic.csv · 파이 크기는 인원수를 뜻하지 않으며, 표본 크기는 각 패널의 '탑승 n명'으로 확인할 것",
        color=c["muted"],
        fontsize=9,
    )
    return fig, tbl


def main() -> None:
    parser = argparse.ArgumentParser(description="성별 × 객실 등급 생존 비율 파이 차트")
    parser.add_argument("--dark", action="store_true", help="다크 모드로 렌더링")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로")
    args = parser.parse_args()

    setup_font()
    c = DARK if args.dark else LIGHT
    fig, tbl = build_figure(dark=args.dark)
    out = args.out or HERE / ("titanic_sex_class_pie_dark.png" if args.dark else "titanic_sex_class_pie.png")
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)

    print(f"저장 완료: {out}")
    print("\n[표 보기] 성별 × 객실 등급 생존 비율")
    print(f"{'집단':<12} {'전체':>6} {'생존':>6} {'사망':>6} {'생존율':>8}")
    print("-" * 44)
    for sex in SEXES:
        for cls in CLASSES:
            r = tbl.loc[(sex, cls)]
            print(
                f"{sex + ' ' + cls:<12} {int(r['전체']):>6} {int(r['생존']):>6} "
                f"{int(r['사망']):>6} {r['생존율']:>7.1f}%"
            )


if __name__ == "__main__":
    main()
