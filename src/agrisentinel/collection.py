"""
src/agrisentinel/collection.py — snapshot-pinned data collection.

Lesson #3 (see PROJECT_STRUCTURE_BOOK.md §0): FAOSTAT revises its bulk file in
place with no version number. Fetching "latest" makes every result silently
unreproducible. The rule enforced here: a snapshot, once pinned, is read-only
and dated; the pipeline defaults to it and only fetches fresh data when
explicitly told to (and even then, it writes a NEW dated folder, never
overwrites an existing one).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import paths

FAO_URL = ("https://bulks-faostat.fao.org/production/"
           "Food_Security_Data_E_All_Data_(Normalized).zip")

# Exact FAOSTAT item names. Never keyword-match these — a keyword match once
# silently pulled the wrong indicator in an earlier project.
FAO_ITEMS = {
    "Prevalence of undernourishment (percent) (3-year average)": "pou",
    "Average dietary energy supply adequacy (percent) (3-year average)": "des_adequacy",
    "Cereal import dependency ratio (percent) (3-year average)": "cereal_import_dep",
    "Per capita food supply variability (kcal/cap/day)": "food_prod_var",
}

WB_INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "SP.POP.GROW": "pop_growth",
}
WB_URL_TMPL = "https://api.worldbank.org/v2/en/indicator/{code}?downloadformat=csv"


@dataclass
class SnapshotResult:
    snapshot_dir: Path
    manifest: dict


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(snap_dir: Path, source_url: str, files: dict[str, bytes], extra: dict) -> dict:
    manifest = {
        "source_url": source_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": _sha256(content), "bytes": len(content)}
                  for name, content in files.items()},
        **extra,
    }
    with open(snap_dir / paths.MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def resolve_faostat_snapshot(snapshot_date: str | None, require_pinned: bool) -> Path:
    """Return the FAOSTAT snapshot dir to read from. Never downloads here —
    that is a separate, explicit step (download_faostat_snapshot)."""
    if snapshot_date:
        d = paths.snapshot_dir(paths.RAW_FAOSTAT_DIR, snapshot_date)
        if not d.exists():
            raise FileNotFoundError(
                f"Pinned FAOSTAT snapshot not found: {d}. "
                f"Run download_faostat_snapshot() once to create it.")
        return d
    existing = sorted(p for p in paths.RAW_FAOSTAT_DIR.glob("*") if p.is_dir())
    if existing:
        return existing[-1]
    if require_pinned:
        raise FileNotFoundError(
            "No pinned FAOSTAT snapshot exists and settings.yaml requires one "
            "(reproducibility.require_pinned_snapshot: true). "
            "Call download_faostat_snapshot() explicitly to create the first one.")
    raise FileNotFoundError("No FAOSTAT snapshot found and pinning is not required — "
                             "nothing to fall back to. Call download_faostat_snapshot().")


def download_faostat_snapshot(snapshot_date: str | None = None, timeout: int = 180) -> SnapshotResult:
    """Download the FAOSTAT bulk file and pin it to a NEW dated folder.
    Refuses to overwrite an existing snapshot for the same date."""
    snap_dir = paths.snapshot_dir(paths.RAW_FAOSTAT_DIR, snapshot_date)
    if snap_dir.exists() and any(snap_dir.iterdir()):
        raise FileExistsError(f"Snapshot already pinned at {snap_dir} — "
                               f"refusing to overwrite. Delete it manually if you really mean to.")
    snap_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(FAO_URL, timeout=timeout)
    resp.raise_for_status()
    zip_bytes = resp.content
    zip_path = snap_dir / "Food_Security_Data_E_All_Data_Normalized.zip"
    zip_path.write_bytes(zip_bytes)

    # Diagnose before trusting: UTF-8 first (latin-1 corrupts names like
    # "Türkiye" -> "TÃ¼rkiye", which then silently fails ISO3 matching later).
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
    raw_bytes = zf.read(csv_name)
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", low_memory=False)
        encoding_used = "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", low_memory=False)
        encoding_used = "latin-1 (fallback)"

    turkiye_present = any("rkiye" in str(a) for a in df["Area"].unique())
    items_found = {name: bool((df["Item"] == name).any()) for name in FAO_ITEMS}

    manifest = _write_manifest(
        snap_dir, FAO_URL, {csv_name: raw_bytes},
        extra={
            "encoding_used": encoding_used,
            "area_count": int(df["Area"].nunique()),
            "row_count": int(len(df)),
            "turkiye_present": turkiye_present,
            "items_found": items_found,
        },
    )
    if not turkiye_present:
        raise AssertionError("Türkiye missing after load — encoding or source problem. "
                              "Do not proceed with this snapshot.")
    if not all(items_found.values()):
        missing = [k for k, v in items_found.items() if not v]
        raise AssertionError(f"Expected FAOSTAT items missing: {missing}")

    return SnapshotResult(snap_dir, manifest)


def load_faostat_raw(snapshot_date: str | None = None, require_pinned: bool = True) -> pd.DataFrame:
    """Load the FAOSTAT raw table from a pinned snapshot (UTF-8 first, as recorded
    in that snapshot's manifest — never re-guessed at read time)."""
    snap_dir = resolve_faostat_snapshot(snapshot_date, require_pinned)
    manifest = json.loads((snap_dir / paths.MANIFEST_FILENAME).read_text())
    csv_name = next(iter(manifest["files"]))
    zip_path = snap_dir / "Food_Security_Data_E_All_Data_Normalized.zip"
    zf = zipfile.ZipFile(zip_path)
    raw_bytes = zf.read(csv_name)
    encoding = "utf-8" if manifest["encoding_used"] == "utf-8" else "latin-1"
    return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, low_memory=False)


def download_worldbank_snapshot(snapshot_date: str | None = None, timeout: int = 120) -> SnapshotResult:
    """Download the 4 World Bank indicators + country classification, pinned."""
    snap_dir = paths.snapshot_dir(paths.RAW_WORLDBANK_DIR, snapshot_date)
    if snap_dir.exists() and any(snap_dir.iterdir()):
        raise FileExistsError(f"Snapshot already pinned at {snap_dir} — refusing to overwrite.")
    snap_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, bytes] = {}
    counts = {}
    for code, name in WB_INDICATORS.items():
        resp = requests.get(WB_URL_TMPL.format(code=code), timeout=timeout)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        data_name = next(n for n in zf.namelist() if n.startswith("API_"))
        meta_name = next((n for n in zf.namelist() if n.startswith("Metadata_Country")), None)
        data_bytes = zf.read(data_name)
        (snap_dir / f"wb_{name}.csv").write_bytes(data_bytes)
        files[f"wb_{name}.csv"] = data_bytes
        if meta_name and not (snap_dir / "country_classification.csv").exists():
            meta_bytes = zf.read(meta_name)
            (snap_dir / "country_classification.csv").write_bytes(meta_bytes)
            files["country_classification.csv"] = meta_bytes
        df = pd.read_csv(io.BytesIO(data_bytes), skiprows=4)
        counts[name] = df.shape

    manifest = _write_manifest(snap_dir, WB_URL_TMPL, files, extra={"shapes": {k: list(v) for k, v in counts.items()}})
    return SnapshotResult(snap_dir, manifest)


def load_worldbank_raw(snapshot_date: str | None = None, require_pinned: bool = True) -> dict[str, pd.DataFrame]:
    """Return {indicator_name: dataframe} plus 'country_classification'."""
    if snapshot_date:
        snap_dir = paths.snapshot_dir(paths.RAW_WORLDBANK_DIR, snapshot_date)
    else:
        existing = sorted(p for p in paths.RAW_WORLDBANK_DIR.glob("*") if p.is_dir())
        if not existing:
            if require_pinned:
                raise FileNotFoundError(
                    "No pinned World Bank snapshot exists. Call download_worldbank_snapshot() first.")
            raise FileNotFoundError("No World Bank snapshot found.")
        snap_dir = existing[-1]
    if not snap_dir.exists():
        raise FileNotFoundError(f"Pinned World Bank snapshot not found: {snap_dir}")

    out = {}
    for name in WB_INDICATORS.values():
        out[name] = pd.read_csv(snap_dir / f"wb_{name}.csv", skiprows=4)
    out["country_classification"] = pd.read_csv(snap_dir / "country_classification.csv")
    return out
