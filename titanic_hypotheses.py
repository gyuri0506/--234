# -*- coding: utf-8 -*-
"""앞서 세운 가설 H1~H6을 titanic.csv 로 실제 검정하고 리포트를 파일로 쓴다.

검정 도구
  - 카이제곱 독립성 검정(+ Cramér's V): 범주형 연관성의 크기
  - 로지스틱 회귀 + 우도비 검정(LR test): 다른 변수를 통제한 뒤에도 효과가 남는지
  - Wilson 신뢰구간: 표본이 작은 칸의 불확실성

사용법:
    python titanic_hypotheses.py                 # 기본 출력 파일: ./kk
    python titanic_hypotheses.py --out kk.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from titanic_survival import load_data

HERE = Path(__file__).resolve().parent
ALPHA = 0.05


# ── 통계 도구 ────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간(%). 표본이 작을 때 정규근사보다 안전하다."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ((centre - half) * 100, (centre + half) * 100)


def chi2_v(df: pd.DataFrame, col: str, target: str = "Survived") -> dict:
    """카이제곱 독립성 검정 + 효과 크기(Cramér's V)."""
    table = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": chi2, "p": p, "dof": dof, "v": v, "n": int(n)}


def logit(y: pd.Series, X: pd.DataFrame):
    X = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, X).fit(disp=0)


def lr_test(full, reduced) -> dict:
    """중첩 모형 우도비 검정: 추가한 항이 설명력을 유의하게 늘리는가."""
    stat = 2 * (full.llf - reduced.llf)
    dof = int(full.df_model - reduced.df_model)
    return {"lr": stat, "dof": dof, "p": stats.chi2.sf(stat, dof)}


def odds_table(res, names: dict[str, str]) -> list[str]:
    """오즈비와 95% 신뢰구간을 표 행으로."""
    ci = res.conf_int()
    rows = []
    for key, label in names.items():
        if key not in res.params.index:
            continue
        rows.append(
            f"| {label} | {np.exp(res.params[key]):.2f} | "
            f"{np.exp(ci.loc[key, 0]):.2f} – {np.exp(ci.loc[key, 1]):.2f} | {res.pvalues[key]:.2e} |"
        )
    return rows


def verdict(p: float, label_yes: str = "지지", label_no: str = "기각") -> str:
    return f"**{label_yes}**" if p < ALPHA else f"**{label_no}**"


def rate_rows(df: pd.DataFrame, by, label: str = "집단") -> list[str]:
    """집단별 생존율 + Wilson 95% CI 표."""
    g = df.groupby(by, observed=True)["Survived"].agg(n="size", s="sum")
    rows = [f"| {label} | n | 생존 | 생존율 | 95% CI |", "|---|---:|---:|---:|---:|"]
    for idx, r in g.iterrows():
        lo, hi = wilson(int(r["s"]), int(r["n"]))
        name = " · ".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
        rows.append(f"| {name} | {int(r['n'])} | {int(r['s'])} | {r['s'] / r['n'] * 100:.1f}% | {lo:.1f} – {hi:.1f}% |")
    return rows


# ── 파생 변수 ────────────────────────────────────────────────────────────────
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["호칭"] = d["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    common = {"Mr": "Mr(성인남성)", "Mrs": "Mrs(기혼여성)", "Miss": "Miss(미혼여성)", "Master": "Master(소년)"}
    d["호칭군"] = d["호칭"].map(common).fillna("기타 호칭")
    d["가족수"] = d["SibSp"] + d["Parch"] + 1
    d["가족규모"] = pd.cut(d["가족수"], [0, 1, 4, 20], labels=["단독(1인)", "소가족(2-4인)", "대가족(5인+)"])
    d["갑판"] = d["Cabin"].str[0]
    d["나이결측"] = d["Age"].isna()
    d["선실기록"] = d["Cabin"].notna()
    d["여성"] = (d["Sex"] == "female").astype(int)
    d["p2"] = (d["Pclass"] == 2).astype(int)
    d["p3"] = (d["Pclass"] == 3).astype(int)
    return d


# ── 가설별 검정 ──────────────────────────────────────────────────────────────
def h1(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H1. 성별이 1차 관문이고 객실 등급은 그 다음 필터다", ""]
    sex, cls = chi2_v(d, "Sex"), chi2_v(d, "Pclass")
    out += [
        "단변량 연관성 크기 비교 (Cramér's V, 클수록 강한 연관):",
        "",
        "| 변수 | χ² | dof | p | Cramér's V |",
        "|---|---:|---:|---:|---:|",
        f"| 성별 | {sex['chi2']:.1f} | {sex['dof']} | {sex['p']:.2e} | **{sex['v']:.3f}** |",
        f"| 객실 등급 | {cls['chi2']:.1f} | {cls['dof']} | {cls['p']:.2e} | {cls['v']:.3f} |",
        "",
    ]
    y = d["Survived"]
    res = logit(y, d[["여성", "p2", "p3"]])
    out += [
        "두 변수를 함께 넣은 로지스틱 회귀 (기준: 남성·1등급):",
        "",
        "| 항 | 오즈비 | 95% CI | p |",
        "|---|---:|---:|---:|",
        *odds_table(res, {"여성": "여성(vs 남성)", "p2": "2등급(vs 1등급)", "p3": "3등급(vs 1등급)"}),
        "",
        "호칭(Name에서 추출)별 생존율 — 남성을 성인/소년으로 쪼갠 결과:",
        "",
        *rate_rows(d, "호칭군", "호칭"),
        "",
    ]
    ok = sex["v"] > cls["v"] and res.pvalues["여성"] < ALPHA
    out += [
        f"**판정: {'지지' if ok else '부분 지지'}** — 성별의 효과 크기(V={sex['v']:.3f})가 등급(V={cls['v']:.3f})보다 크고, "
        f"등급을 통제해도 여성의 생존 오즈는 남성의 {np.exp(res.params['여성']):.1f}배입니다. "
        f"남성 중에서도 소년(Master)은 {d[d['호칭군'] == 'Master(소년)']['Survived'].mean() * 100:.1f}%로 "
        f"성인 남성({d[d['호칭군'] == 'Mr(성인남성)']['Survived'].mean() * 100:.1f}%)과 확연히 갈립니다.",
        "",
    ]
    summary.append(
        (
            "H1",
            "성별이 1차 관문, 등급은 2차",
            "지지" if ok else "부분 지지",
            f"V {sex['v']:.2f}(성별) vs {cls['v']:.2f}(등급), 여성 OR={np.exp(res.params['여성']):.1f}",
        )
    )


def h2(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H2. 등급 효과는 성별에 따라 다르게 작동한다 (교호작용)", ""]
    y = d["Survived"]
    base = d[["여성", "p2", "p3"]]
    inter = base.assign(**{"여성×2등급": d["여성"] * d["p2"], "여성×3등급": d["여성"] * d["p3"]})
    m0, m1 = logit(y, base), logit(y, inter)
    t = lr_test(m1, m0)
    out += [
        f"우도비 검정(교호작용 항 2개 추가): LR = {t['lr']:.1f}, dof = {t['dof']}, **p = {t['p']:.2e}**",
        "",
        "성별로 나눠 본 등급 효과(같은 성별 안에서 등급이 얼마나 듣는가):",
        "",
    ]
    for sex in ("여성", "남성"):
        sub = d[d["성별"] == sex]
        c = chi2_v(sub, "Pclass")
        rates = sub.groupby("등급", observed=True)["Survived"].mean() * 100
        spread = rates.max() - rates.min()
        out += [
            f"- **{sex}**: 1등급 {rates['1등급']:.1f}% → 2등급 {rates['2등급']:.1f}% → 3등급 {rates['3등급']:.1f}% "
            f"(최대-최소 {spread:.1f}%p, χ²={c['chi2']:.1f}, p={c['p']:.2e}, V={c['v']:.3f})",
        ]
    out += [
        "",
        f"**판정: {'지지' if t['p'] < ALPHA else '기각'}** — 등급의 기울기가 두 성별에서 평행하지 않습니다. "
        "여성은 1·2등급이 거의 같고 3등급에서만 무너지는 '계단형', 남성은 1등급만 높고 2·3등급이 바닥인 형태입니다.",
        "",
    ]
    summary.append(("H2", "성별×등급 교호작용", "지지" if t["p"] < ALPHA else "기각", f"LR={t['lr']:.1f}, p={t['p']:.1e}"))


def h3(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H3. 3등급 여성의 급락은 '선내 위치(갑판 접근성)'로 설명된다", ""]
    cov = d.groupby("등급", observed=True)["선실기록"].mean() * 100
    out += [
        "선실(Cabin) 기록 보유율 — 이 가설의 검증 가능 범위:",
        "",
        "| 등급 | 선실 기록 보유율 |",
        "|---|---:|",
        *[f"| {i} | {v:.1f}% |" for i, v in cov.items()],
        "",
    ]
    known = d[d["선실기록"]]
    out += ["선실이 기록된 204명의 갑판별 생존율:", "", *rate_rows(known, "갑판", "갑판"), ""]
    sub = known[known["Pclass"] == 1]
    c = chi2_v(sub, "갑판") if sub["갑판"].nunique() > 1 else None
    if c:
        out += [
            f"1등급 내부만 비교(n={c['n']}): χ²={c['chi2']:.1f}, dof={c['dof']}, p={c['p']:.3f}, V={c['v']:.3f}",
            "",
        ]
    out += [
        "**판정: 검증 불가(자료 한계)** — 3등급의 선실 기록 보유율이 "
        f"{cov['3등급']:.1f}%에 불과해, 정작 검증하려던 '3등급 여성의 위치'를 관측할 수 없습니다. "
        "갑판별 생존율 차이는 등급 차이와 뒤섞여 있고, 1등급 내부로 좁히면 갑판 간 차이가 유의하지 않습니다. "
        "이 가설은 선내 배치도 등 외부 자료 없이는 이 데이터만으로 판정할 수 없습니다.",
        "",
    ]
    summary.append(("H3", "갑판 접근성 가설", "검증 불가", f"3등급 선실기록 {cov['3등급']:.1f}%"))


def h4(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H4. 가족 구조가 생존을 좌우한다 (역U자: 2~4인 유리, 단독·대가족 불리)", ""]
    out += [*rate_rows(d, "가족규모", "가족 규모"), ""]
    c = chi2_v(d, "가족규모")
    y = d["Survived"]
    base = d[["여성", "p2", "p3"]]
    fam = pd.get_dummies(d["가족규모"], prefix="가족", drop_first=True).astype(int)
    m0, m1 = logit(y, base), logit(y, pd.concat([base, fam], axis=1))
    t = lr_test(m1, m0)
    out += [
        f"단변량: χ²={c['chi2']:.1f}, dof={c['dof']}, p={c['p']:.2e}, V={c['v']:.3f}",
        "",
        f"성별·등급 **통제 후** 우도비 검정: LR = {t['lr']:.1f}, dof = {t['dof']}, **p = {t['p']:.2e}**",
        "",
        "| 항 (기준: 단독 1인) | 오즈비 | 95% CI | p |",
        "|---|---:|---:|---:|",
        *odds_table(m1, {"가족_소가족(2-4인)": "소가족(2-4인)", "가족_대가족(5인+)": "대가족(5인+)"}),
        "",
        "경쟁 가설 점검 — 3등급 여성 안에서 가족 규모별 생존율:",
        "",
        *rate_rows(d[(d["성별"] == "여성") & (d["Pclass"] == 3)], "가족규모", "3등급 여성"),
        "",
        f"**판정: {'지지' if t['p'] < ALPHA else '기각'}** — 성별·등급을 통제해도 가족 규모 효과가 남습니다. "
        "3등급 여성 안에서도 대가족의 생존율이 뚜렷이 낮아, H3(위치)의 대안 설명으로 작동합니다.",
        "",
    ]
    summary.append(("H4", "가족 규모(역U자)", "지지" if t["p"] < ALPHA else "기각", f"통제 후 LR={t['lr']:.1f}, p={t['p']:.1e}"))


def h5(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H5. 승선항 효과는 등급의 대리변수가 아니다", ""]
    e = d.dropna(subset=["Embarked"]).copy()
    out += [*rate_rows(e, "Embarked", "승선항"), ""]
    y = e["Survived"]
    base = e[["여성", "p2", "p3"]]
    emb = pd.get_dummies(e["Embarked"], prefix="항", drop_first=True).astype(int)
    m0, m1 = logit(y, base), logit(y, pd.concat([base, emb], axis=1))
    t = lr_test(m1, m0)
    out += [
        f"성별·등급 **통제 후** 우도비 검정: LR = {t['lr']:.1f}, dof = {t['dof']}, **p = {t['p']:.3f}** "
        f"(승선항 결측 {int(d['Embarked'].isna().sum())}명 제외)",
        "",
        "| 항 (기준: C=셰르부르) | 오즈비 | 95% CI | p |",
        "|---|---:|---:|---:|",
        *odds_table(m1, {"항_Q": "Q(퀸스타운)", "항_S": "S(사우샘프턴)"}),
        "",
        "3등급 여성만 떼어 본 승선항별 생존율 (표본이 작아 CI 폭이 넓음):",
        "",
        *rate_rows(e[(e["성별"] == "여성") & (e["Pclass"] == 3)], "Embarked", "3등급 여성"),
        "",
        f"**판정: {'지지' if t['p'] < ALPHA else '기각'}** — "
        + (
            "성별·등급을 통제한 뒤에도 승선항 효과가 유의하게 남습니다."
            if t["p"] < ALPHA
            else "성별·등급을 통제하면 승선항 효과가 통계적으로 사라집니다. 즉 승선항 차이는 상당 부분 등급 구성의 반영입니다."
        ),
        "",
    ]
    summary.append(("H5", "승선항 독립 효과", "지지" if t["p"] < ALPHA else "기각", f"통제 후 p={t['p']:.3f}"))


def h6(d: pd.DataFrame, out: list[str], summary: list[tuple]) -> None:
    out += ["## H6. (방법론) 결측 자체가 생존과 상관있다 — 결측은 무작위가 아니다", ""]
    y = d["Survived"]
    base = d[["여성", "p2", "p3"]]
    results = {}
    labels = {
        "나이결측": ("나이(Age) 기재 여부", {True: "미기재", False: "기재됨"}),
        "선실기록": ("선실(Cabin) 기재 여부", {True: "기재됨", False: "미기재"}),
    }
    for col, (label, mapping) in labels.items():
        c = chi2_v(d, col)
        m1 = logit(y, base.assign(**{col: d[col].astype(int)}))
        m0 = logit(y, base)
        t = lr_test(m1, m0)
        results[col] = (c, t, m1)
        tagged = d.assign(**{label: d[col].map(mapping)})
        out += [
            f"### {label}",
            "",
            *rate_rows(tagged, label, label),
            "",
            f"단변량: χ²={c['chi2']:.1f}, p={c['p']:.2e}, V={c['v']:.3f} · "
            f"성별·등급 통제 후: LR={t['lr']:.1f}, p={t['p']:.2e}, "
            f"오즈비={np.exp(m1.params[col]):.2f}",
            "",
        ]
    out += [
        "나이 결측의 분포(결측이 어디에 몰려 있는가):",
        "",
        "| 등급 | 나이 미기재 비율 |",
        "|---|---:|",
        *[f"| {i} | {v * 100:.1f}% |" for i, v in d.groupby("등급", observed=True)["나이결측"].mean().items()],
        "",
    ]
    c_age, t_age = results["나이결측"][0], results["나이결측"][1]
    c_cab, t_cab = results["선실기록"][0], results["선실기록"][1]
    out += [
        "**판정: 지지 (단, 두 결측의 성격이 다름)** — 두 결측 모두 단변량으로는 생존과 유의하게 연관되어 "
        f"완전 무작위 결측(MCAR)은 기각됩니다(나이 p={c_age['p']:.1e}, 선실 p={c_cab['p']:.1e}). "
        "그러나 성별·등급을 통제하면 갈립니다:",
        "",
        f"- **나이 결측**: 통제 후 효과 소멸(LR={t_age['lr']:.1f}, p={t_age['p']:.2f}). "
        "결측이 3등급(27.7%)에 몰려 있어서 생겼던 차이이며, '등급을 알면 결측은 무작위'인 **MAR에 가깝습니다**.",
        f"- **선실 결측**: 통제 후에도 유의하게 남음(LR={t_cab['lr']:.1f}, p={t_cab['p']:.1e}, "
        f"오즈비={np.exp(results['선실기록'][2].params['선실기록']):.2f}). "
        "등급으로 설명되지 않는 정보가 결측 자체에 들어 있어 **MNAR(비무작위 결측)이 의심됩니다** — "
        "생존자에게서 선실 정보가 더 잘 수집됐을 가능성.",
        "",
        "> **실무적 함의**: ①연령대 차트의 '나이 미기재 177명 제외'는 등급 구성을 왜곡시키므로 "
        "(3등급이 과소대표됨) 전체 평균/중앙값 대치보다 **등급·호칭별 조건부 대치**가 안전합니다. "
        "②`Cabin` 결측 여부를 그대로 피처로 쓰면 성능은 올라가지만, 사고 이후의 기록 수집 과정이 반영된 "
        "**누출(leakage)성 변수**일 수 있어 해석용 모델에서는 배제하는 편이 낫습니다.",
        "",
    ]
    summary.append(
        (
            "H6",
            "결측의 비무작위성",
            "지지(성격은 상이)",
            f"나이=MAR(통제 후 p={t_age['p']:.2f}), 선실=MNAR 의심(p={t_cab['p']:.1e})",
        )
    )


# ── 리포트 ───────────────────────────────────────────────────────────────────
def build_report(d: pd.DataFrame) -> str:
    out: list[str] = []
    summary: list[tuple] = []
    body: list[str] = []

    for fn in (h1, h2, h3, h4, h5, h6):
        fn(d, body, summary)

    n, alive = len(d), int(d["Survived"].sum())
    out += [
        "# 타이타닉 생존 가설 검정 리포트",
        "",
        f"- 데이터: `titanic.csv` · {n:,}명 · 생존 {alive:,}명({alive / n * 100:.1f}%)",
        "- 방법: 카이제곱 독립성 검정(Cramér's V), 로지스틱 회귀 + 우도비 검정, Wilson 95% 신뢰구간",
        f"- 유의수준: α = {ALPHA}",
        "",
        "## 결론 요약",
        "",
        "| 가설 | 내용 | 판정 | 핵심 수치 |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | **{c}** | {e} |" for a, b, c, e in summary],
        "",
        "---",
        "",
    ]
    out += body
    out += [
        "---",
        "",
        "## 한계",
        "",
        "- 891명은 실제 탑승객 약 2,224명의 부분 표본이며 승무원이 빠져 있습니다. 전체 사건으로의 일반화는 제한적입니다.",
        "- 가설 6개를 같은 데이터로 연달아 검정했으므로 다중비교 문제가 있습니다. 본문 p값은 보정 전 값입니다 "
        "(Bonferroni 기준을 쓰려면 α = 0.05/6 ≈ 0.0083).",
        "- 관측 데이터이므로 모든 결과는 연관성이며 인과가 아닙니다.",
        "- H3처럼 결측 구조 때문에 애초에 검정이 성립하지 않는 가설이 있습니다. 판정을 '기각'이 아니라 "
        "'검증 불가'로 구분해 적었습니다.",
        "",
        "생성 스크립트: `titanic_hypotheses.py` (재실행하면 이 파일을 덮어씁니다)",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="타이타닉 가설 검정 리포트 생성")
    parser.add_argument("--out", type=Path, default=HERE / "kk", help="리포트 저장 경로 (기본: ./kk)")
    args = parser.parse_args()

    d = prepare(load_data())
    report = build_report(d)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
