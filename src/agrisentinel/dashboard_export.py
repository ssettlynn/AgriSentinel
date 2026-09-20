"""
src/agrisentinel/dashboard_export.py — THE single writer of
site/agrisentinel_data.js. Lesson #1 (PROJECT_STRUCTURE_BOOK.md §0): a UI
that showed 34/168/96/2160 for weeks because those numbers were typed
directly into HTML. The rule this module enforces: no other file may write
this JSON, and this module never invents a number — every value here is
either read straight from an already-computed metrics dict/dataframe or is a
plain aggregation over SERIES-level data. If a number can't be traced to an
upstream artifact, it does not belong in this file.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import paths

NICE_LABELS = {
    "pou": "Hunger level", "des_adequacy": "Food supply adequacy",
    "cereal_import_dep": "Import dependence", "food_prod_var": "Supply stability",
    "gdp_per_capita": "Income per person", "gdp_growth": "Economic growth",
    "inflation_cpi": "Inflation", "pop_growth": "Population growth",
    "pou_change": "Hunger momentum", "covid_flag": "Pandemic year",
}
UNITS = {"pou": "%", "gdp_per_capita": "$", "inflation_cpi": "%",
         "des_adequacy": "%", "pop_growth": "%", "gdp_growth": "%",
         "cereal_import_dep": "%", "food_prod_var": "kcal"}


def _round(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def build_series_and_meta(master_with_targets: pd.DataFrame) -> tuple[dict, dict]:
    """Per-country time series (for charts) + display metadata."""
    series, meta = {}, {}
    indicator_cols = ["pou", "des_adequacy", "cereal_import_dep", "food_prod_var",
                       "gdp_per_capita", "gdp_growth", "inflation_cpi", "pop_growth"]
    for iso, g in master_with_targets.sort_values("year").groupby("country_iso"):
        series[iso] = {"y": g["year"].tolist(),
                        **{col: [_round(v) for v in g[col]] for col in indicator_cols},
                        "tier": [None if pd.isna(t) else t for t in g["risk_tier_current"]]}
        first = g.iloc[0]
        meta[iso] = {"n": first["country_name"], "r": first["region"], "g": first["income_group"]}
    return series, meta


def build_global_aggregates(master_with_targets: pd.DataFrame) -> dict:
    """Global trend, per-region trend, per-income-group 2023 median — powers
    the descriptive World & Patterns charts, computed live, not hardcoded."""
    years = sorted(master_with_targets["year"].unique().tolist())
    global_pou = master_with_targets.groupby("year")["pou"].mean().reindex(years).round(2).tolist()

    regions = {}
    for region, g in master_with_targets.groupby("region"):
        regions[region] = g.groupby("year")["pou"].mean().reindex(years).round(2).tolist()

    latest_year = int(master_with_targets["year"].max())
    latest = master_with_targets[master_with_targets["year"] == latest_year]
    income_latest = latest.groupby("income_group")["pou"].median().round(1).to_dict()

    return {"years": years, "global_pou": global_pou, "regions": regions,
            f"income_{latest_year}": income_latest, "latest_year": latest_year}


def build_home_stats(master_with_targets: pd.DataFrame, pred_next_year: dict,
                      tier_change: dict) -> dict:
    """Every number the Home page shows, aggregated live from the same data
    everything else reads — never a separately hand-typed figure."""
    total_countries = int(master_with_targets["country_iso"].nunique())
    high_next_year = sum(1 for p in pred_next_year.values() if p.get("rf", {}).get("t") == "High")
    agree = sum(1 for p in pred_next_year.values()
                if len({p.get(k, {}).get("t") for k in ("rf", "lr", "dt")}) == 1)
    agree_pct = round(agree / len(pred_next_year) * 100) if pred_next_year else 0

    recent_shock_year = int(master_with_targets["year"].max()) - 1
    shock_high = int((master_with_targets[master_with_targets.year == recent_shock_year]["pou"] >= 15).sum())

    return {
        "countries": total_countries,
        "obs": int(len(master_with_targets)),
        "high_next_year": high_next_year,
        "agree_pct": agree_pct,
        "high_recent_shock_year": {"year": recent_shock_year, "count": shock_high},
        "tier_change": tier_change,
        "forecast_coverage": len(pred_next_year),
    }


def build_predictions(demo_latest: pd.DataFrame, fitted_tier_1yr: dict[str, Pipeline],
                       feat10: list[str]) -> dict:
    """Next-year tier prediction per country, all three 1-year models."""
    out: dict[str, dict] = {}
    for kind, key in [("logreg", "lr"), ("dtree", "dt"), ("rf", "rf")]:
        model = fitted_tier_1yr[kind]
        classes = list(model.named_steps["clf"].classes_)
        hi = classes.index("High")
        pred = model.predict(demo_latest[feat10])
        proba = model.predict_proba(demo_latest[feat10])
        for i, iso in enumerate(demo_latest["country_iso"].values):
            out.setdefault(iso, {})[key] = {
                "t": pred[i], "c": _round(proba[i].max()), "pH": _round(proba[i][hi]),
            }
    return out


def build_outlook(demo_latest: pd.DataFrame, fitted_tier_5yr_rf: Pipeline,
                   feat10: list[str], target_year: int) -> dict:
    """5-year-ahead tier forecast per country, from the long-horizon RF."""
    pred = fitted_tier_5yr_rf.predict(demo_latest[feat10])
    proba = fitted_tier_5yr_rf.predict_proba(demo_latest[feat10])
    out = {}
    for i, iso in enumerate(demo_latest["country_iso"].values):
        out.setdefault(iso, {})[str(target_year)] = {"t": pred[i], "c": _round(proba[i].max(), 2)}
    return out


def build_direction_predictions(demo_latest: pd.DataFrame, fitted_direction: dict[str, Pipeline],
                                 feat14: list[str]) -> tuple[dict, dict]:
    dir1, dir5 = {}, {}
    for horizon, target_dict in [("1yr", dir1), ("5yr", dir5)]:
        model = fitted_direction[horizon]
        pred = model.predict(demo_latest[feat14])
        proba = model.predict_proba(demo_latest[feat14])
        for i, iso in enumerate(demo_latest["country_iso"].values):
            target_dict[iso] = {"d": pred[i], "c": _round(proba[i].max())}
    return dir1, dir5


def build_logreg_export(fitted_logreg: Pipeline, feat_list: list[str]) -> dict:
    """Serializes a fitted logreg Pipeline (imputer + scaler + classifier) into
    plain numbers so the What-If Lab can re-run the exact same softmax
    client-side, live, with no server round-trip. This is the one place
    model internals cross into the UI — every number here is read straight
    off the fitted sklearn objects, never re-derived by hand."""
    imp, sc, clf = fitted_logreg.named_steps["imp"], fitted_logreg.named_steps["sc"], fitted_logreg.named_steps["clf"]
    return {
        "features": feat_list,
        "medians": [round(float(v), 4) for v in imp.statistics_],
        "mean": [round(float(v), 4) for v in sc.mean_],
        "scale": [round(float(v), 4) for v in sc.scale_],
        "coef": [[round(float(w), 5) for w in row] for row in clf.coef_],
        "intercept": [round(float(v), 5) for v in clf.intercept_],
        "classes": list(clf.classes_),
    }


def build_indicator_forecasts(demo_latest: pd.DataFrame,
                               fitted_forecasters: dict[tuple[str, str], Pipeline],
                               indicators: list[str], feat14: list[str]) -> dict:
    """Per-country, per-indicator {now, +1yr, +5yr} — powers the Forecast /
    What-If indicator tiles."""
    out: dict[str, dict] = {}
    for i, row in demo_latest.iterrows():
        iso = row["country_iso"]
        out[iso] = {}
        for ind in indicators:
            entry = {"now": _round(row[ind], 2)}
            for h in ("1yr", "5yr"):
                model = fitted_forecasters.get((ind, h))
                if model is not None:
                    pred = model.predict(demo_latest.loc[[i], feat14])[0]
                    entry[f"y{h}"] = _round(pred, 2)
            out[iso][ind] = entry
    return out


def build_missing_values(master: pd.DataFrame, feats: list[str]) -> list[dict]:
    """Missing-value rate per indicator on the merged dataset, before any
    imputation — the Project Book's missing-values table. Reported here rather
    than recomputed in the browser so the site and the book cannot drift."""
    n = len(master)
    out = []
    for f in feats:
        if f not in master.columns:
            continue
        miss = int(master[f].isna().sum())
        out.append({"indicator": f, "missing": miss,
                    "pct": _round(miss / n * 100, 2) if n else 0.0,
                    "present": n - miss})
    return sorted(out, key=lambda r: -r["missing"])


def build_evaluation(views: dict[str, pd.DataFrame], fitted_rf: dict[str, Pipeline],
                     feats: list[str], targets: dict[str, str]) -> dict:
    """The honest-evaluation payload: confusion matrix, per-class scores, the
    validation threshold sweep, and bootstrap intervals against the persistence
    baseline — the Chapter 4 evidence, computed on the TEST split only.

    These already exist as functions in `modeling`; this assembles them for the
    dashboard so the site shows the same evidence as the Project Book instead of
    asserting results whose working is invisible.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    from . import modeling

    tiers = ["Low", "Medium", "High"]
    out: dict[str, dict] = {}
    for h, df in views.items():
        target = targets[h]
        test = df[df.split == "test"]
        rf = fitted_rf[h]
        pred = rf.predict(test[feats])

        cm = confusion_matrix(test[target], pred, labels=tiers)
        rep = classification_report(test[target], pred, labels=tiers,
                                    output_dict=True, zero_division=0)
        # persistence = "same tier as this year", the baseline every score is judged against
        base_pred = test["risk_tier_current"].to_numpy()
        entry = {
            "tiers": tiers,
            "confusion": cm.tolist(),
            "n_test": int(len(test)),
            "per_class": {t: {"precision": _round(rep[t]["precision"]),
                              "recall": _round(rep[t]["recall"]),
                              "f1": _round(rep[t]["f1-score"]),
                              "support": int(rep[t]["support"])} for t in tiers},
            "rf_ci": modeling.bootstrap_ci_macro_f1(test[target], pred),
            "base_ci": modeling.bootstrap_ci_macro_f1(test[target], base_pred),
        }
        entry["ci_overlap"] = not (entry["rf_ci"]["ci_hi"] < entry["base_ci"]["ci_lo"]
                                   or entry["base_ci"]["ci_hi"] < entry["rf_ci"]["ci_lo"])

        # threshold sweep is a VALIDATION-set exercise; the chosen cut is then
        # applied once to test, which is the number worth reporting
        x_val, y_val = modeling._split_xy(df, "val", target, feats)
        sweep = modeling.threshold_sweep(rf, x_val, y_val)
        chosen = modeling.choose_threshold(sweep)
        classes = list(rf.named_steps["clf"].classes_)
        hi = classes.index("High")
        proba = rf.predict_proba(test[feats])[:, hi]
        pred_bin = (proba >= chosen).astype(int)
        true_bin = (test[target] == "High").astype(int)
        tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
        entry["threshold"] = {
            "sweep": sweep.to_dict(orient="records"),
            "chosen": chosen,
            "test_recall": _round(tp / max(int(true_bin.sum()), 1)),
            "test_precision": _round(tp / max(int(pred_bin.sum()), 1)),
        }
        out[h] = entry
    return out


def write_dashboard_data(
    series: dict, meta: dict, glob: dict, home_stats: dict,
    tasks: list[dict], metrics: list[dict], rules: pd.DataFrame,
    shap_1yr: dict, shap_5yr: dict, predictions: dict, outlook: dict,
    dir1: dict, dir5: dict, dir_metrics: list[dict],
    indicator_forecasts: dict, forecast_metrics: list[dict],
    logreg_1yr: dict, logreg_5yr: dict, local_shap: dict | None = None,
    data_stats: dict | None = None, missing_values: list[dict] | None = None,
    evaluation: dict | None = None, fbeta: list[dict] | None = None,
    class_weights: list[dict] | None = None,
) -> None:
    """The ONLY function in the whole project allowed to write
    site/agrisentinel_data.js. Every argument here must already be a plain
    dict/list/dataframe produced by an earlier pipeline step — this function
    does no computation of its own beyond serialization."""
    payload = {
        "SERIES": series, "META": meta, "GLOB": glob, "HOMESTATS": home_stats,
        "TASKS": tasks, "METRICS": metrics,
        "RULES": rules.to_dict(orient="records"),
        "SHAP": {"1yr": shap_1yr, "5yr": shap_5yr},
        # Per-country signed SHAP for the 1-year RF, from
        # explainability.local_shap_by_country(). SHAP above is the global mean
        # (one ranking for every country); this is per prediction, so the
        # country page can answer "why THIS country" and not only "what matters
        # on average". Optional: an older caller that does not pass it still
        # writes a valid file, the UI just falls back to the global view.
        "SHAPLOCAL": local_shap or {},
        "PRED": predictions, "OUTLOOK": outlook,
        "DIR": dir1, "DIR5": dir5, "DIRMETRICS": dir_metrics,
        "FORE": indicator_forecasts, "FCMETRICS": forecast_metrics,
        "LOGREG": logreg_1yr, "LOGREG5": logreg_5yr,
        # The descriptive-analysis payload: data_statistics.json exactly as the
        # statistics stage computed it (summary stats, IQR outliers, Pearson vs
        # Spearman, normality, ANOVA/Kruskal group tests, tier balance) plus the
        # missing-value table. Passed through verbatim so the Descriptive page
        # shows the same numbers as the Project Book, never a re-derivation.
        "DATASTATS": data_stats or {},
        "MISSING": missing_values or [],
        # Chapter 4 evidence: confusion matrix, per-class scores, threshold sweep
        # and bootstrap intervals — from build_evaluation(), test split only.
        "EVAL": evaluation or {},
        # F-beta at several values of beta, from modeling.fbeta_comparison().
        # Macro-F1 (the METRICS table) weights precision and recall equally;
        # this shows how the ranking changes once a missed High-risk country is
        # weighted more heavily than a false alarm. beta = 1 reproduces macro-F1.
        "FBETA": fbeta or [],
        # class_weight sweep from modeling.class_weight_comparison(): what
        # raising the weight on the High tier buys (recall) and costs
        # (precision), per model family. Validation split — never test.
        "CWEIGHTS": class_weights or [],
        "NICE": NICE_LABELS, "UNIT": UNITS,
    }
    paths.SITE_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.DASHBOARD_DATA_JS, "w") as f:
        f.write("const AGRISENTINEL_DATA = ")
        json.dump(payload, f, allow_nan=False, default=str)
        f.write(";\n")
