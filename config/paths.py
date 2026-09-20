"""
config/paths.py — THE single source of truth for every path in AgriSentinel.

Rule (see PROJECT_STRUCTURE_BOOK.md §1): no other file — notebook, script, test,
or doc — may spell out a path as a string literal. Everything imports these
constants. If a folder or filename needs to change, it changes here once, and
every consumer (Colab notebook, desktop project, dashboard exporter, tests)
picks it up automatically.

config/schema.json is a language-agnostic mirror of this file, for any future
non-Python consumer (e.g. a JS build step). tests/test_dashboard_contract.py
asserts the two stay in sync.
"""
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data — raw (pinned, dated snapshots; never overwritten, never "latest")
# ---------------------------------------------------------------------------
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
RAW_FAOSTAT_DIR    = RAW_DIR / "faostat"
RAW_WORLDBANK_DIR  = RAW_DIR / "worldbank"

def snapshot_dir(source_dir: Path, snapshot_date: str | None = None) -> Path:
    """Return the pinned snapshot folder for a source, e.g. data/raw/faostat/2026-07-28/.
    Defaults to today only when creating a NEW snapshot — reading code should
    always pass an explicit date (usually the one recorded in manifest.json)
    so a stale re-run can never silently pick up a different day's data."""
    d = snapshot_date or date.today().isoformat()
    return source_dir / d

MANIFEST_FILENAME = "manifest.json"   # {sha256, row_count, area_count, source_url, retrieved_at}

# ---------------------------------------------------------------------------
# Data — interim & processed (regenerable, gitignored)
# ---------------------------------------------------------------------------
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"

MASTER_DATASET          = PROCESSED_DIR / "master_dataset.csv"
FEATURES_1YR            = PROCESSED_DIR / "features_lagged_1yr.csv"
FEATURES_5YR            = PROCESSED_DIR / "features_lagged_5yr.csv"
TRANSACTIONS_APRIORI    = PROCESSED_DIR / "transactions_apriori.csv"
FEATURES_DEMO_LATEST    = PROCESSED_DIR / "features_demo_latest.csv"   # most recent complete year, for live forecasts

# ---------------------------------------------------------------------------
# Models (versioned by training-run date)
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"

def models_run_dir(run_date: str | None = None) -> Path:
    d = run_date or date.today().isoformat()
    return MODELS_DIR / d

MODEL_FILENAMES = {
    "logreg_1yr": "logreg_1yr.pkl", "logreg_5yr": "logreg_5yr.pkl",
    "dtree_1yr":  "dtree_1yr.pkl",  "dtree_5yr":  "dtree_5yr.pkl",
    "rf_1yr":     "rf_best_1yr.pkl", "rf_5yr":    "rf_best_5yr.pkl",
    "direction_1yr": "direction_1yr.pkl", "direction_5yr": "direction_5yr.pkl",
    "forecast_prefix": "forecast_{indicator}_{horizon}.pkl",   # 5 indicators x 2 horizons
    "config": "model_config.pkl",
}

# ---------------------------------------------------------------------------
# Reports (machine-readable — every doc claim must cite one of these keys)
# ---------------------------------------------------------------------------
REPORTS_DIR      = PROJECT_ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"
METRICS_DIR      = REPORTS_DIR / "metrics"

METRICS_JSON            = METRICS_DIR / "metrics.json"             # tier scoreboard, both horizons
DATA_STATISTICS_JSON    = METRICS_DIR / "data_statistics.json"      # descriptive stats (§Data Statistics)
FORECAST_METRICS_JSON   = METRICS_DIR / "forecast_metrics.json"     # multi-indicator R²/MAE
RULES_JSON              = METRICS_DIR / "rules.json"                # Apriori association rules
SHAP_JSON               = METRICS_DIR / "shap.json"                 # feature importance, both horizons
DIRECTION_METRICS_JSON  = METRICS_DIR / "direction_metrics.json"     # direction task scoreboard

# ---------------------------------------------------------------------------
# Site (static UI) — the ONE generated data file the UI is allowed to read numbers from
# ---------------------------------------------------------------------------
SITE_DIR         = PROJECT_ROOT / "site"
SITE_ASSETS_DIR  = SITE_DIR / "assets"
DASHBOARD_DATA_JS = SITE_DIR / "agrisentinel_data.js"

SITE_PAGES = ["index.html", "explore.html", "country.html", "forecast.html",
              "modellab.html", "lab.html", "learn.html"]

# ---------------------------------------------------------------------------
# Notebook / src / docs / tests
# ---------------------------------------------------------------------------
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PIPELINE_NOTEBOOK = NOTEBOOKS_DIR / "agrisentinel_pipeline.ipynb"

SRC_DIR  = PROJECT_ROOT / "src" / "agrisentinel"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------------
# Run parameters that affect paths/schema (kept here, not scattered in notebook cells)
# ---------------------------------------------------------------------------
YEAR_MIN, YEAR_MAX = 2010, 2023
HORIZONS = ("1yr", "5yr")
TASKS = ("tier", "value", "direction", "forecast")
INDICATORS = ("pou", "gdp_per_capita", "inflation_cpi", "des_adequacy", "pop_growth")


def ensure_project_dirs() -> None:
    """Create every directory this schema declares. Safe to call repeatedly.
    This is the ONLY function allowed to call Path.mkdir for project folders —
    notebook cells and scripts call this instead of mkdir-ing ad hoc paths."""
    for d in (RAW_FAOSTAT_DIR, RAW_WORLDBANK_DIR, INTERIM_DIR, PROCESSED_DIR,
              MODELS_DIR, FIGURES_DIR, METRICS_DIR, SITE_ASSETS_DIR,
              NOTEBOOKS_DIR, SRC_DIR, DOCS_DIR, TESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Self-check (lesson #5): print the resolved root so a misconfigured
    # working directory is caught immediately, not silently.
    print("PROJECT_ROOT resolved to:", PROJECT_ROOT)
    ensure_project_dirs()
    print("All schema directories present.")
