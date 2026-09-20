"""
Tests for the teacher-suggested statistical add-ons:
  statistics.group_significance / correlation_gap / normality_report
  modeling.imbalance_experiment / flag_ablation / ols_regression

These check the CONTRACT (keys, shapes, value ranges) and the core logic each
function is supposed to prove, on a small synthetic panel built so the intended
signal is present: three regions with clearly different hunger levels, a strictly
linear driver (des_adequacy), and a curved driver (gdp_per_capita). Kept
synthetic and small so the suite stays fast and never depends on regenerable
data files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agrisentinel import modeling, statistics

FEATURES = ["pou", "des_adequacy", "cereal_import_dep", "food_prod_var",
            "gdp_per_capita", "gdp_growth", "inflation_cpi", "pop_growth"]


def _synthetic_master(per_region: int = 60) -> pd.DataFrame:
    """Three regions at Low / Medium / High hunger, with one linear and one
    curved relationship to pou planted on purpose."""
    rng = np.random.default_rng(0)
    blocks = []
    specs = [("Region-Low", 1.0, 4.0), ("Region-Med", 6.0, 14.0), ("Region-High", 16.0, 40.0)]
    for region, lo, hi in specs:
        pou = rng.uniform(lo, hi, per_region)
        df = pd.DataFrame({
            "country_iso": [f"{region[:3].upper()}{i}" for i in range(per_region)],
            "country_name": region,
            "year": rng.integers(2010, 2024, per_region),
            "region": region,
            "income_group": rng.choice(
                ["Low income", "Lower middle income", "Upper middle income", "High income"], per_region),
            "pou": pou,
            "des_adequacy": 140.0 - 1.5 * pou + rng.normal(0, 2, per_region),      # strong linear
            "cereal_import_dep": rng.uniform(0, 90, per_region),
            "food_prod_var": rng.uniform(10, 60, per_region),
            "gdp_per_capita": 50000.0 * np.exp(-pou / 8.0) + rng.normal(0, 500, per_region),  # curved
            "gdp_growth": rng.normal(2, 3, per_region),
            "inflation_cpi": rng.uniform(0, 20, per_region),
            "pop_growth": rng.normal(1.5, 1.0, per_region),
        })
        blocks.append(df)
    master = pd.concat(blocks, ignore_index=True)
    for c in FEATURES:                       # audit-trail flag columns, all "reported"
        master[c + "_filled"] = 0
    return master


# --------------------------- statistics --------------------------------------

def test_group_significance_detects_real_region_difference():
    res = statistics.group_significance(_synthetic_master(), by="region", value="pou")
    assert set(res) >= {"anova_F", "anova_p", "kruskal_H", "kruskal_p", "significant", "group_medians"}
    assert res["n_groups"] == 3
    assert res["significant"] is True
    assert res["kruskal_p"] < 0.05 and res["anova_p"] < 0.05


def test_correlation_gap_shape_and_linear_driver():
    gaps = statistics.correlation_gap(_synthetic_master(), target="pou")
    assert "pou" not in gaps and len(gaps) == 7
    for v in gaps.values():
        assert {"pearson", "spearman", "gap"} <= set(v)
    # des_adequacy is planted as a strong linear driver
    assert gaps["des_adequacy"]["pearson"] < -0.7
    # the curved driver's monotonic (spearman) strength exceeds its linear one
    assert gaps["gdp_per_capita"]["gap"] > 0


def test_normality_report_flags_skewed_feature():
    rep = statistics.normality_report(_synthetic_master())
    assert len(rep) == 8
    for v in rep.values():
        assert {"skew", "normaltest_p", "normal_at_5pct"} <= set(v)
        assert isinstance(v["skew"], float)
    # the exp-distributed driver is clearly non-normal, and pou is right-skewed
    assert rep["gdp_per_capita"]["normal_at_5pct"] is False
    assert rep["pou"]["skew"] > 0.5


# --------------------------- modeling ----------------------------------------

def test_imbalance_experiment_contract():
    out = modeling.imbalance_experiment(_synthetic_master(), n_splits=3)
    assert list(out["setting"]) == ["unbalanced", "balanced"]
    for col in ("macro_f1", "high_recall"):
        assert out[col].between(0.0, 1.0).all()


def test_flag_ablation_all_zero_flags_do_not_help():
    out = modeling.flag_ablation(_synthetic_master(), n_estimators=40, n_splits=3)
    assert set(out) >= {"features_only_macro_f1", "features_plus_flags_macro_f1",
                        "delta", "flags_help", "n_flags"}
    assert out["n_flags"] == 8
    assert 0.0 <= out["features_only_macro_f1"] <= 1.0
    assert out["flags_help"] is False                   # constant flags carry no signal
    assert out["delta"] <= 0.01


def test_fbeta_at_beta_one_reproduces_macro_f1():
    """beta = 1 is F1 by definition, so the F-beta table must agree exactly with
    macro-F1 — this is what makes the two scoreboards comparable."""
    from sklearn.metrics import f1_score
    y_true = ["Low", "Low", "Medium", "High", "High", "High"]
    y_pred = ["Low", "Medium", "Medium", "High", "High", "Medium"]
    table = modeling.fbeta_table(y_true, {"model": y_pred}, betas=(1.0,))
    expected = round(float(f1_score(y_true, y_pred, average="macro")), 3)
    assert table.loc[0, "macro_fbeta"] == expected


def test_fbeta_rises_with_beta_when_recall_beats_precision():
    """A model that over-predicts High has recall 1.0 but precision below 1.
    Raising beta weights recall more, so its High F-beta must increase."""
    y_true = ["Low", "Low", "High", "High"]
    y_pred = ["High", "Low", "High", "High"]      # High: recall 1.0, precision 2/3
    table = modeling.fbeta_table(y_true, {"model": y_pred}, betas=(0.5, 1.0, 2.0))
    v = table.set_index("beta")["high_fbeta"]
    assert v[0.5] < v[1.0] < v[2.0]


def test_class_weight_grid_covers_none_balanced_and_custom():
    """The sweep must include the two reference settings plus custom weights,
    and every custom weight must name all three tiers or sklearn will raise."""
    labels = [label for label, _ in modeling.CLASS_WEIGHT_GRID]
    assert "none" in labels and "balanced" in labels
    customs = [w for _, w in modeling.CLASS_WEIGHT_GRID if isinstance(w, dict)]
    assert customs, "the grid should contain at least one explicit weighting"
    for w in customs:
        assert set(w) == {"Low", "Medium", "High"}
        assert w["High"] > w["Low"], "High is the tier the weighting is meant to favour"


def test_ols_regression_reports_significant_linear_driver():
    res = modeling.ols_regression(_synthetic_master(), target="pou")
    ct = res["coef_table"]
    assert list(ct["term"])[0] == "const" and len(ct) == 8
    assert 0.0 < res["r2"] <= 1.0
    p_des = float(ct.loc[ct["term"] == "des_adequacy", "p_value"].iloc[0])
    assert p_des < 0.05                                  # the planted driver is significant
