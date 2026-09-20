#!/usr/bin/env python3
"""
scripts/run_pipeline.py — run the AgriSentinel pipeline from PyCharm (or any
terminal), with zero duplicated logic versus the Colab notebook.

This script calls the exact same functions in src/agrisentinel/ that
notebooks/agrisentinel_pipeline.ipynb calls. Nothing here re-implements
cleaning, feature engineering, modeling, or export — it only orchestrates and
logs. That is the concrete proof of PROJECT_STRUCTURE_BOOK.md's "one pipeline,
one place" rule: a bug fixed in src/agrisentinel/ is fixed for the notebook
AND the desktop project at once, because there is only one implementation.

Usage:
    python scripts/run_pipeline.py                  # run every stage
    python scripts/run_pipeline.py --stage collect   # run one stage only
    python scripts/run_pipeline.py --list-stages     # show available stages

After `pip install -e .`, this is also available as the `agrisentinel-run`
command (see pyproject.toml).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# macOS-only: NumPy 2.x raises spurious floating-point flags from BLAS matmul/vecdot
# under Apple's Accelerate backend. They are false positives — the numeric results are
# correct and the same code is clean on Colab/Linux — so they are silenced by exact
# message, never as a blanket filter, so any real numeric warning still surfaces.
for _msg in ("divide by zero encountered in matmul", "overflow encountered in matmul",
             "invalid value encountered in matmul", "divide by zero encountered in vecdot",
             "overflow encountered in vecdot", "invalid value encountered in vecdot"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import paths, settings
from src.agrisentinel import (collection, cleaning, features, statistics,
                               modeling, explainability, dashboard_export)

log = logging.getLogger("agrisentinel")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def stage_collect(s: dict) -> None:
    """Collect (or reuse the pinned snapshot of) both data sources."""
    snap_date = s["reproducibility"]["snapshot_date"]
    require_pinned = s["reproducibility"]["require_pinned_snapshot"]

    try:
        fao_raw = collection.load_faostat_raw(snap_date, require_pinned)
        log.info("FAOSTAT: loaded pinned snapshot (%s) — %s rows", snap_date, len(fao_raw))
    except FileNotFoundError:
        log.warning("No pinned FAOSTAT snapshot — downloading and pinning a new one.")
        result = collection.download_faostat_snapshot(snap_date)
        log.info("FAOSTAT: pinned at %s (%s areas)", result.snapshot_dir, result.manifest["area_count"])

    try:
        collection.load_worldbank_raw(snap_date, require_pinned)
        log.info("World Bank: loaded pinned snapshot (%s)", snap_date)
    except FileNotFoundError:
        log.warning("No pinned World Bank snapshot — downloading and pinning a new one.")
        result = collection.download_worldbank_snapshot(snap_date)
        log.info("World Bank: pinned at %s", result.snapshot_dir)


def stage_clean(s: dict) -> None:
    """Build and save the master panel."""
    snap_date = s["reproducibility"]["snapshot_date"]
    require_pinned = s["reproducibility"]["require_pinned_snapshot"]

    fao_raw = collection.load_faostat_raw(snap_date, require_pinned)
    wb_frames = collection.load_worldbank_raw(snap_date, require_pinned)
    master = cleaning.build_master_dataset(fao_raw, wb_frames)
    master.to_csv(paths.MASTER_DATASET, index=False)
    log.info("master_dataset.csv: %s | countries: %s", master.shape, master["country_iso"].nunique())


def _load_master_with_targets():
    """Shared by every downstream stage — read master, engineer features, add targets."""
    import pandas as pd
    master = pd.read_csv(paths.MASTER_DATASET)
    m = features.add_engineered_features(master)
    m = features.add_targets(m)
    return master, m


def stage_statistics(s: dict) -> None:
    """Descriptive report -> reports/metrics/data_statistics.json."""
    master, m = _load_master_with_targets()
    data_stats = {
        "coverage": statistics.coverage_summary(master),
        "descriptive": statistics.descriptive_statistics(master),
        "outliers_iqr": statistics.outliers_iqr(master),
        "correlation_with_pou": statistics.correlation_with_target(master),
        "correlation_gap": statistics.correlation_gap(master),
        "normality": statistics.normality_report(master),
        "group_significance": {
            "region": statistics.group_significance(master, by="region"),
            "income_group": statistics.group_significance(master, by="income_group"),
        },
        "tier_distribution": statistics.tier_distribution(m),
    }
    with open(paths.DATA_STATISTICS_JSON, "w") as f:
        json.dump(data_stats, f, indent=2, default=str)
    log.info("saved %s", paths.DATA_STATISTICS_JSON)


def stage_features(s: dict) -> tuple:
    """Build the four model-ready views."""
    _, m = _load_master_with_targets()
    d1yr = features.build_lagged_view(m, "1yr")
    d5yr = features.build_lagged_view(m, "5yr")
    tx = features.build_transactions(m)
    demo_latest = features.build_demo_latest(m)

    d1yr.to_csv(paths.FEATURES_1YR, index=False)
    d5yr.to_csv(paths.FEATURES_5YR, index=False)
    tx.to_csv(paths.TRANSACTIONS_APRIORI, index=False)
    demo_latest.to_csv(paths.FEATURES_DEMO_LATEST, index=False)
    log.info("1yr view: %s | 5yr view: %s | transactions: %s | demo: %s",
              d1yr.shape, d5yr.shape, tx.shape, demo_latest.shape)
    return m, d1yr, d5yr, tx, demo_latest


def stage_mine_rules(s: dict) -> None:
    """Apriori association rules -> reports/metrics/rules.json."""
    _, _, _, tx, _ = stage_features(s)
    rules = explainability.mine_association_rules(tx, min_support=0.05, min_lift=1.2)
    rules.to_json(paths.RULES_JSON, orient="records", indent=2)
    log.info("mined %s rules -> %s", len(rules), paths.RULES_JSON)


def stage_train(s: dict) -> None:
    """Train every model (tier, direction, forecast) and save artifacts + metrics."""
    import joblib
    m, d1yr, d5yr, tx, demo_latest = stage_features(s)

    log.info("training tier classifiers (Persistence/LogReg/DTree/RF, both horizons)...")
    fitted_tier, scoreboard, _ = modeling.train_tier_models(d1yr, d5yr)
    with open(paths.METRICS_JSON, "w") as f:
        json.dump(scoreboard, f, indent=2, default=str)

    # Both extra metric files are written beside metrics.json rather than given
    # their own path constants, so config/paths.py and config/schema.json stay in
    # sync (tests/test_paths_schema_sync.py enforces that).

    # class_weight sweep — how much weight the High tier should carry. Scored on
    # VALIDATION only, since class_weight is a modelling choice, not a result.
    cw_df = modeling.class_weight_comparison(d1yr, d5yr)
    cw_path = paths.METRICS_JSON.parent / "class_weights.json"
    cw_df.to_json(cw_path, orient="records", indent=2)
    log.info("class_weight sweep: %s rows (%s settings) -> %s",
             len(cw_df), len(modeling.CLASS_WEIGHT_GRID), cw_path)

    # F-beta on the same fitted models and the same TEST rows as the scoreboard.
    fbeta_df = modeling.fbeta_comparison(d1yr, d5yr, fitted_tier)
    fbeta_path = paths.METRICS_JSON.parent / "fbeta.json"
    fbeta_df.to_json(fbeta_path, orient="records", indent=2)
    log.info("F-beta comparison: %s rows (betas %s) -> %s",
             len(fbeta_df), list(modeling.FBETA_VALUES), fbeta_path)

    run_dir = paths.models_run_dir(s["reproducibility"]["snapshot_date"])
    run_dir.mkdir(parents=True, exist_ok=True)
    for (horizon, kind), pipe in fitted_tier.items():
        key = {"logreg": "logreg", "dtree": "dtree", "rf": "rf_best"}[kind]
        joblib.dump(pipe, run_dir / f"{key}_{horizon}.pkl")
    joblib.dump({"features": features.FEAT10}, run_dir / "model_config.pkl")

    for h in ("1yr", "5yr"):
        rf_test = next(r for r in scoreboard if r["horizon"] == h and r["model"] == "Random Forest" and r["split"] == "test")
        log.info("%s Random Forest test macro-F1: %s", h, rf_test["macro_f1"])

    log.info("training direction classifiers (both horizons)...")
    fitted_direction, direction_metrics, _dir_grids = modeling.train_direction_models(m)
    with open(paths.DIRECTION_METRICS_JSON, "w") as f:
        json.dump(direction_metrics, f, indent=2, default=str)
    for pipe_h, pipe in fitted_direction.items():
        joblib.dump(pipe, run_dir / f"direction_{pipe_h}.pkl")
    for row in direction_metrics:
        log.info("%s direction macro-F1: %s (baseline %s)", row["horizon"], row["macro_f1"], row["baseline_macro_f1"])

    log.info("training multi-indicator forecast regressors (5 indicators x 2 horizons)...")
    indicators = ["pou", "gdp_per_capita", "inflation_cpi", "des_adequacy", "pop_growth"]
    fitted_forecast, forecast_metrics = modeling.train_forecast_regressors(m, indicators)
    with open(paths.FORECAST_METRICS_JSON, "w") as f:
        json.dump(forecast_metrics, f, indent=2)
    for (indicator, horizon), pipe in fitted_forecast.items():
        joblib.dump(pipe, run_dir / f"forecast_{indicator}_{horizon}.pkl")

    log.info("computing SHAP global importance, both horizons...")
    shap_1yr = explainability.global_shap_importance(fitted_tier[("1yr", "rf")], d1yr, "risk_tier_next1")
    shap_5yr = explainability.global_shap_importance(fitted_tier[("5yr", "rf")], d5yr, "risk_tier_next5")
    with open(paths.SHAP_JSON, "w") as f:
        json.dump({"1yr": shap_1yr, "5yr": shap_5yr}, f, indent=2)

    log.info("all models trained and saved to %s", run_dir)


def stage_export(s: dict) -> None:
    """Assemble every model + metric artifact and write the single dashboard JSON."""
    import joblib
    m, d1yr, d5yr, tx, demo_latest = stage_features(s)
    run_dir = paths.models_run_dir(s["reproducibility"]["snapshot_date"])

    def _load(key, h=None):
        name = f"{key}_{h}.pkl" if h else f"{key}.pkl"
        return joblib.load(run_dir / name)

    fitted_tier_1yr = {"logreg": _load("logreg", "1yr"), "dtree": _load("dtree", "1yr"), "rf": _load("rf_best", "1yr")}
    fitted_tier_5yr_rf = _load("rf_best", "5yr")
    fitted_direction = {"1yr": _load("direction", "1yr"), "5yr": _load("direction", "5yr")}
    indicators = ["pou", "gdp_per_capita", "inflation_cpi", "des_adequacy", "pop_growth"]
    fitted_forecast = {(ind, h): _load(f"forecast_{ind}", h) for ind in indicators for h in ("1yr", "5yr")}

    scoreboard = json.loads(paths.METRICS_JSON.read_text())
    direction_metrics = json.loads(paths.DIRECTION_METRICS_JSON.read_text())
    forecast_metrics = json.loads(paths.FORECAST_METRICS_JSON.read_text())
    shap_data = json.loads(paths.SHAP_JSON.read_text())
    import pandas as pd
    rules = pd.read_json(paths.RULES_JSON)

    series, meta = dashboard_export.build_series_and_meta(m)
    glob = dashboard_export.build_global_aggregates(m)
    predictions = dashboard_export.build_predictions(demo_latest, fitted_tier_1yr, features.FEAT10)
    latest_year = int(m["year"].max())
    outlook = dashboard_export.build_outlook(demo_latest, fitted_tier_5yr_rf, features.FEAT10, latest_year + 5)
    dir1, dir5 = dashboard_export.build_direction_predictions(demo_latest, fitted_direction, features.FEAT14)
    indicator_forecasts = dashboard_export.build_indicator_forecasts(demo_latest, fitted_forecast, indicators, features.FEAT14)

    tier_change = {
        "1yr": round(float((d1yr["risk_tier_current"] != d1yr["risk_tier_next1"]).mean() * 100), 1),
        "5yr": round(float((d5yr["risk_tier_current"] != d5yr["risk_tier_next5"]).mean() * 100), 1),
    }
    home_stats = dashboard_export.build_home_stats(m, predictions, tier_change)

    rf1 = next(r for r in scoreboard if r["horizon"] == "1yr" and r["model"] == "Random Forest" and r["split"] == "test")
    base1 = next(r for r in scoreboard if r["horizon"] == "1yr" and r["model"] == "Persistence" and r["split"] == "test")
    pou_fc = next(r for r in forecast_metrics if r["indicator"] == "pou" and r["horizon"] == "1yr")
    tasks = [
        {"name": "Risk tier", "type": "Classification", "note": "Low / Medium / High next year",
         "score": rf1["macro_f1"], "baseline": base1["macro_f1"], "metric": "macro-F1"},
        {"name": "Hunger level", "type": "Regression", "note": "The actual undernourishment %",
         "score": pou_fc["mae"], "baseline": pou_fc["baseline_mae"], "metric": "MAE (pp)"},
        {"name": "Risk direction", "type": "Classification",
         "note": "Improving / stable / worsening — the hardest of the three",
         "score": direction_metrics[0]["macro_f1"], "baseline": direction_metrics[0]["baseline_macro_f1"],
         "metric": "macro-F1"},
    ]

    logreg_1yr = dashboard_export.build_logreg_export(fitted_tier_1yr["logreg"], features.FEAT10)
    logreg_5yr_model = _load("logreg", "5yr")
    logreg_5yr = dashboard_export.build_logreg_export(logreg_5yr_model, features.FEAT10)

    # Per-country "why this tier" for the same 1-year RF whose prediction the
    # country page shows. Computed here (not inside dashboard_export) because
    # write_dashboard_data does serialization only — every value it writes is
    # produced by an earlier step.
    local_shap = explainability.local_shap_by_country(
        fitted_tier_1yr["rf"], demo_latest, features.FEAT10)
    log.info("local SHAP computed for %d countries", len(local_shap))

    # descriptive analysis: reuse the statistics stage's own output rather than
    # recomputing, so the site cannot disagree with the Project Book
    data_stats = json.loads(paths.DATA_STATISTICS_JSON.read_text())
    missing_values = dashboard_export.build_missing_values(m, features.FEATURES_RAW)
    evaluation = dashboard_export.build_evaluation(
        {"1yr": d1yr, "5yr": d5yr},
        {"1yr": fitted_tier_1yr["rf"], "5yr": fitted_tier_5yr_rf},
        features.FEAT10,
        {"1yr": "risk_tier_next1", "5yr": "risk_tier_next5"},
    )

    fbeta_path = paths.METRICS_JSON.parent / "fbeta.json"
    fbeta = json.loads(fbeta_path.read_text()) if fbeta_path.exists() else []
    cw_path = paths.METRICS_JSON.parent / "class_weights.json"
    class_weights = json.loads(cw_path.read_text()) if cw_path.exists() else []

    dashboard_export.write_dashboard_data(
        series, meta, glob, home_stats, tasks, scoreboard, rules,
        shap_data["1yr"], shap_data["5yr"], predictions, outlook, dir1, dir5, direction_metrics,
        indicator_forecasts, forecast_metrics, logreg_1yr, logreg_5yr,
        local_shap=local_shap,
        data_stats=data_stats, missing_values=missing_values,
        evaluation=evaluation, fbeta=fbeta, class_weights=class_weights,
    )
    log.info("wrote %s (%s bytes)", paths.DASHBOARD_DATA_JS, paths.DASHBOARD_DATA_JS.stat().st_size)
    log.info("home stats: %s", home_stats)


STAGES = {
    "collect": stage_collect,
    "clean": stage_clean,
    "statistics": stage_statistics,
    "features": lambda s: stage_features(s) and None,
    "mine_rules": stage_mine_rules,
    "train": stage_train,
    "export": stage_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgriSentinel pipeline.")
    parser.add_argument("--stage", choices=list(STAGES), default=None,
                        help="run one stage only (default: run every stage in order)")
    parser.add_argument("--list-stages", action="store_true", help="print available stages and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_stages:
        for name in STAGES:
            print(name)
        return

    _configure_logging(args.verbose)
    paths.ensure_project_dirs()
    s = settings.get_settings()
    log.info("PROJECT_ROOT: %s", paths.PROJECT_ROOT)
    log.info("pinned snapshot date: %s (require_pinned=%s)",
              s["reproducibility"]["snapshot_date"], s["reproducibility"]["require_pinned_snapshot"])

    order = [args.stage] if args.stage else ["collect", "clean", "statistics", "mine_rules", "train", "export"]
    for stage_name in order:
        log.info("=" * 60)
        log.info("STAGE: %s", stage_name)
        log.info("=" * 60)
        STAGES[stage_name](s)
    log.info("pipeline complete.")


if __name__ == "__main__":
    main()
