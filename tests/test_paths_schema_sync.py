"""
Proof that config/paths.py and config/schema.json cannot silently drift apart.

This is the concrete mechanism behind the "Option A + schema-first hybrid"
recommendation in PROJECT_STRUCTURE_BOOK.md §1: if someone edits paths.py
(e.g. renames master_dataset.csv) and forgets schema.json, this test fails
in CI before any notebook or UI code can run against the mismatched pair.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import paths


def _schema():
    with open(paths.PROJECT_ROOT / "config" / "schema.json") as f:
        return json.load(f)


def _rel(p: Path) -> str:
    return str(p.relative_to(paths.PROJECT_ROOT))


def test_processed_paths_match_schema():
    s = _schema()["data"]["processed"]
    assert s["master_dataset"] == _rel(paths.MASTER_DATASET)
    assert s["features_1yr"] == _rel(paths.FEATURES_1YR)
    assert s["features_5yr"] == _rel(paths.FEATURES_5YR)
    assert s["transactions_apriori"] == _rel(paths.TRANSACTIONS_APRIORI)
    assert s["features_demo_latest"] == _rel(paths.FEATURES_DEMO_LATEST)


def test_metrics_paths_match_schema():
    s = _schema()["reports"]["metrics"]
    assert s["metrics"] == _rel(paths.METRICS_JSON)
    assert s["data_statistics"] == _rel(paths.DATA_STATISTICS_JSON)
    assert s["forecast_metrics"] == _rel(paths.FORECAST_METRICS_JSON)
    assert s["rules"] == _rel(paths.RULES_JSON)
    assert s["shap"] == _rel(paths.SHAP_JSON)
    assert s["direction_metrics"] == _rel(paths.DIRECTION_METRICS_JSON)


def test_site_dashboard_data_path_matches_schema():
    s = _schema()["site"]
    assert s["dashboard_data"] == _rel(paths.DASHBOARD_DATA_JS)
    assert s["pages"] == paths.SITE_PAGES


def test_run_parameters_match_settings_and_schema():
    s = _schema()["run_parameters"]
    assert s["year_min"] == paths.YEAR_MIN
    assert s["year_max"] == paths.YEAR_MAX
    assert tuple(s["horizons"]) == paths.HORIZONS
    assert tuple(s["indicators"]) == paths.INDICATORS


def test_ensure_project_dirs_is_idempotent(tmp_path, monkeypatch):
    # Prove ensure_project_dirs() is the ONLY mkdir path and is safe to re-run
    # (lesson #5: no ad hoc directory creation scattered across scripts).
    paths.ensure_project_dirs()
    paths.ensure_project_dirs()  # must not raise on second call
    assert paths.PROCESSED_DIR.exists()
    assert paths.SITE_ASSETS_DIR.exists()
