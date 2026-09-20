"""
src/agrisentinel/statistics.py — the descriptive layer: coverage, missingness,
distributions, outliers, correlation. This is what reports/metrics/data_statistics.json
is built from, and what §5 EDA charts read alongside SERIES/GLOB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FEATURES_RAW = ["pou", "des_adequacy", "cereal_import_dep", "food_prod_var",
                "gdp_per_capita", "gdp_growth", "inflation_cpi", "pop_growth"]


def coverage_summary(master: pd.DataFrame) -> dict:
    return {
        "rows": int(len(master)),
        "countries": int(master["country_iso"].nunique()),
        "year_min": int(master["year"].min()),
        "year_max": int(master["year"].max()),
        "regions": int(master["region"].nunique()),
        "income_groups": int(master["income_group"].nunique()),
    }


def descriptive_statistics(master: pd.DataFrame) -> dict:
    desc = master[FEATURES_RAW].describe().T
    desc["median"] = master[FEATURES_RAW].median()
    desc["skew"] = master[FEATURES_RAW].skew()
    desc["missing"] = master[FEATURES_RAW].isna().sum()
    desc["missing_pct"] = (desc["missing"] / len(master) * 100).round(2)
    return desc.round(2).to_dict(orient="index")


def outliers_iqr(master: pd.DataFrame, k: float = 1.5) -> dict:
    """1.5x-IQR rule per feature — reported as a fact about the data, not
    treated as errors to remove (Zimbabwe's 557% inflation is real, not noise)."""
    out = {}
    for col in FEATURES_RAW:
        s = master[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        out[col] = {"n_outliers": n_out, "pct": round(n_out / len(s) * 100, 1),
                     "range_min": round(float(s.min()), 1), "range_max": round(float(s.max()), 1)}
    return out


def correlation_with_target(master: pd.DataFrame, target: str = "pou") -> dict:
    others = [c for c in FEATURES_RAW if c != target]
    out = {}
    for col in others:
        pair = master[[target, col]].dropna()
        if len(pair) < 3:
            continue
        pearson = pair[target].corr(pair[col], method="pearson")
        spearman = pair[target].corr(pair[col], method="spearman")
        out[col] = {"pearson": round(float(pearson), 3), "spearman": round(float(spearman), 3)}
    return out


def tier_distribution(master_with_tier: pd.DataFrame) -> dict:
    counts = master_with_tier["risk_tier_current"].value_counts()
    total = counts.sum()
    return {tier: {"count": int(counts.get(tier, 0)), "pct": round(float(counts.get(tier, 0) / total * 100), 1)}
            for tier in ("Low", "Medium", "High")}


# ---------------------------------------------------------------------------
# Inferential add-ons (teacher's suggestions): move beyond describing the data
# to TESTING claims about it — is a group difference real, is a link curved,
# is a feature far enough from normal to need a transform. Each returns a plain
# dict/frame so the notebook, the report, and tests share one implementation.
# ---------------------------------------------------------------------------

def group_significance(master: pd.DataFrame, by: str = "region", value: str = "pou") -> dict:
    """Is the difference in `value` across groups of `by` real, or could it be
    chance? One-way ANOVA tests the means (assumes roughly normal); Kruskal-Wallis
    tests the medians with no normality assumption. Both are reported because
    `pou` is skewed, so the distribution-free Kruskal result is the one to trust."""
    groups = [g[value].dropna().values for _, g in master.groupby(by)]
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kruskal = stats.kruskal(*groups)
    return {
        "by": by, "value": value, "n_groups": len(groups),
        "anova_F": round(float(f_stat), 2), "anova_p": float(p_anova),
        "kruskal_H": round(float(h_stat), 2), "kruskal_p": float(p_kruskal),
        "significant": bool(p_kruskal < 0.05),
        "group_medians": master.groupby(by)[value].median().round(2).sort_values().to_dict(),
    }


def correlation_gap(master: pd.DataFrame, target: str = "pou") -> dict:
    """Pearson (straight-line strength) vs Spearman (any monotonic trend) for
    each feature against `target`. A large positive gap means the relationship is
    monotonic but CURVED — a signal that a linear model will underuse the feature
    while a tree captures it. This is what turns 'look at correlation' into a
    concrete modelling decision."""
    others = [c for c in FEATURES_RAW if c != target]
    out = {}
    for col in others:
        pair = master[[target, col]].dropna()
        if len(pair) < 3:
            continue
        pearson = float(pair[target].corr(pair[col], method="pearson"))
        spearman = float(pair[target].corr(pair[col], method="spearman"))
        out[col] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3),
                    "gap": round(abs(spearman) - abs(pearson), 3)}
    return out


def normality_report(master: pd.DataFrame) -> dict:
    """D'Agostino-Pearson normality test + skew per feature — the evidence base
    for deciding which features need a transform before a LINEAR model. p < 0.05
    rejects normality; a |skew| well above 1 alongside a rejected test flags a
    transform candidate (e.g. log for gdp_per_capita). Tree models need none of
    this, which is itself part of the reasoning for preferring them here."""
    out = {}
    for col in FEATURES_RAW:
        s = master[col].dropna()
        stat, p = stats.normaltest(s)
        out[col] = {"skew": round(float(s.skew()), 2),
                    "normaltest_p": float(p),
                    "normal_at_5pct": bool(p > 0.05)}
    return out
