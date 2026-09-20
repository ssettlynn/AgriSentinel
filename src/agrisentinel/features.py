"""
src/agrisentinel/features.py — engineered features, the three prediction
targets (tier / value / direction), chronological splits, and the Apriori
transaction view. Every function here is pure (dataframe in, dataframe out)
so the notebook, tests, and any future script call the same logic — never a
re-typed copy (lesson #2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import get_settings

FEATURES_RAW = ["pou", "des_adequacy", "cereal_import_dep", "food_prod_var",
                "gdp_per_capita", "gdp_growth", "inflation_cpi", "pop_growth"]

FEAT10 = FEATURES_RAW + ["pou_change", "covid_flag"]                              # classification
FEAT14 = FEAT10 + ["pou_trend3", "des_change", "infl_change", "pou_vs_region"]     # regression / direction


def to_tier(pou: float) -> str | float:
    if pd.isna(pou):
        return np.nan
    return "Low" if pou < 5 else ("Medium" if pou < 15 else "High")


def _slope3(series: pd.Series) -> pd.Series:
    """3-year rolling linear-fit slope: a medium-term trend signal, distinct
    from the single-year pou_change momentum feature."""
    out = series.copy() * np.nan
    for i in range(len(series)):
        window = series.iloc[max(0, i - 2): i + 1].dropna()
        if len(window) >= 2:
            out.iloc[i] = np.polyfit(range(len(window)), window.values, 1)[0]
    return out


def add_engineered_features(master: pd.DataFrame) -> pd.DataFrame:
    """Adds pou_change, covid_flag (base, FEAT10) and pou_trend3, des_change,
    infl_change, pou_vs_region (extended, FEAT14 only). Requires the frame be
    sorted by (country_iso, year) — shift()/diff() depend on row order."""
    m = master.sort_values(["country_iso", "year"]).reset_index(drop=True).copy()
    m["pou_change"] = m.groupby("country_iso")["pou"].diff()
    m["covid_flag"] = m["year"].isin([2020, 2021]).astype(int)

    m["pou_trend3"] = m.groupby("country_iso")["pou"].transform(_slope3)
    m["des_change"] = m.groupby("country_iso")["des_adequacy"].diff()
    m["infl_change"] = m.groupby("country_iso")["inflation_cpi"].diff()
    m["pou_vs_region"] = m["pou"] - m.groupby(["region", "year"])["pou"].transform("mean")

    m["risk_tier_current"] = m["pou"].apply(to_tier)
    return m


def add_targets(master: pd.DataFrame) -> pd.DataFrame:
    """Adds the three prediction targets, both horizons:
      TARGET 1 (tier, classification):    risk_tier_next1 / risk_tier_next5
      TARGET 2 (value, regression):       pou_next1 / pou_next5
      TARGET 3 (direction, classification): dir_next1 / dir_next5
    """
    s = get_settings()["direction_task"]
    m = master.copy()
    g = m.groupby("country_iso")

    m["risk_tier_next1"] = g["risk_tier_current"].shift(-1)
    m["risk_tier_next5"] = g["risk_tier_current"].shift(-5)
    m["pou_next1"] = g["pou"].shift(-1)
    m["pou_next5"] = g["pou"].shift(-5)

    def direction(delta: pd.Series) -> np.ndarray:
        return np.where(delta > s["worsening_threshold_pp"], "Worsening",
                np.where(delta < s["improving_threshold_pp"], "Improving", "Stable"))

    for h in (1, 5):
        delta = m[f"pou_next{h}"] - m["pou"]
        m[f"dir_next{h}"] = direction(delta)
        m.loc[delta.isna(), f"dir_next{h}"] = np.nan

    # Leakage sanity check: next-year tier at t must equal current tier at t+1.
    chk = m[["country_iso", "year", "risk_tier_current", "risk_tier_next1"]].copy()
    chk["cur_next_year"] = chk.groupby("country_iso")["risk_tier_current"].shift(-1)
    bad = chk.dropna(subset=["risk_tier_next1", "cur_next_year"])
    bad = bad[bad["risk_tier_next1"] != bad["cur_next_year"]]
    assert len(bad) == 0, "target misaligned — check sort order before shift()"

    return m


def add_split(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Chronological train/val/test split — never random, so the model can
    never learn from the future and be tested on the past."""
    s = get_settings()["splits"][horizon]
    tr_end, va_end = s["train_end"], s["val_end"]
    out = df.copy()
    out["split"] = np.where(out["year"] <= tr_end, "train",
                     np.where(out["year"] <= va_end, "val", "test"))
    return out


def build_lagged_view(master_with_targets: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """One horizon's model-ready view: rows with a labelled target, split assigned."""
    target_col = f"risk_tier_next{1 if horizon == '1yr' else 5}"
    view = master_with_targets.dropna(subset=[target_col]).copy()
    view = add_split(view, horizon)
    for split in ("train", "val", "test"):
        classes = view.loc[view.split == split, target_col].nunique()
        assert classes == 3, f"{horizon}/{split} split is missing a class ({classes}/3 present)"
    return view


def build_transactions(master_with_targets: pd.DataFrame) -> pd.DataFrame:
    """Binary yes/no items for Apriori. Income tertile cutoffs are computed on
    the TRAIN period only (year <= 1yr train_end), so no future information
    leaks into the mining step.

    Rows with a missing source value are dropped BEFORE the boolean columns
    are built, not after: `NaN < 5` evaluates to False, not NaN, so a
    trailing .dropna() on the boolean columns silently keeps incomplete rows
    with every flag in a group set to False — breaking the mutual-exclusivity
    guarantee below without raising anything on its own."""
    s = get_settings()
    train_end = s["splits"]["1yr"]["train_end"]
    required_cols = ["pou", "gdp_per_capita", "inflation_cpi", "cereal_import_dep",
                      "des_adequacy", "gdp_growth", "pop_growth"]
    m = master_with_targets.dropna(subset=required_cols).copy()
    train_gdp = m[m.year <= train_end]["gdp_per_capita"]
    lo, hi = train_gdp.quantile([1 / 3, 2 / 3])

    tx = pd.DataFrame({
        "country_iso": m["country_iso"], "year": m["year"],
        "HUNGER_LOW": m["pou"] < 5,
        "HUNGER_MED": m["pou"].between(5, 15, inclusive="left"),
        "HUNGER_HIGH": m["pou"] >= 15,
        "INCOME_LOW": m["gdp_per_capita"] < lo,
        "INCOME_MID": m["gdp_per_capita"].between(lo, hi, inclusive="left"),
        "INCOME_HIGH": m["gdp_per_capita"] >= hi,
        "INFLATION_HIGH": m["inflation_cpi"] >= 10,
        "IMPORT_DEP_HIGH": m["cereal_import_dep"] >= 50,
        "SUPPLY_INADEQUATE": m["des_adequacy"] < 100,
        "ECON_SHRINKING": m["gdp_growth"] < 0,
        "POPGROWTH_HIGH": m["pop_growth"] >= 2,
    })

    for group in (["HUNGER_LOW", "HUNGER_MED", "HUNGER_HIGH"], ["INCOME_LOW", "INCOME_MID", "INCOME_HIGH"]):
        assert (tx[group].sum(axis=1) == 1).all(), f"{group} is not mutually exclusive per row"
    return tx


def build_demo_latest(master_with_targets: pd.DataFrame) -> pd.DataFrame:
    """Most recent complete year, no target yet — powers live forecasts on the
    site. Kept file-level separate from training data (never merged in), so a
    future contributor cannot accidentally train on unlabelled rows."""
    latest_year = int(master_with_targets["year"].max())
    return master_with_targets[master_with_targets["year"] == latest_year].copy()
