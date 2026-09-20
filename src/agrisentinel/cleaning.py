"""
src/agrisentinel/cleaning.py — encoding-safe, leakage-free cleaning & integration.

Lesson #4 (PROJECT_STRUCTURE_BOOK.md §0): "China, Taiwan Province of" matches
BOTH the CHN and TWN regex patterns in country_converter, which then returns a
LIST instead of a string — and a list is unhashable, which crashes any later
duplicate-check with "TypeError: unhashable type: 'list'". The fix is a small,
explicit, hand-written override table, checked by tests/test_cleaning.py
against a fixed fixture list before this module is trusted.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import country_converter as coco

from config import paths
from config.settings import get_settings
from src.agrisentinel.collection import FAO_ITEMS

FEATURES_RAW = ["pou", "des_adequacy", "cereal_import_dep", "food_prod_var",
                "gdp_per_capita", "gdp_growth", "inflation_cpi", "pop_growth"]

# Aggregates that are not countries — dropped BEFORE ISO3 conversion, not after,
# because converting an aggregate name can silently produce a real country's code.
DROP_AREAS = {
    "China", "World", "Africa", "Asia", "Europe", "Americas", "Oceania",
    "Southern Africa", "Micronesia", "European Union (27)",
    "Least Developed Countries", "Land Locked Developing Countries",
    "Small Island Developing States", "Low Income Food Deficit Countries",
    "Net Food Importing Developing Countries",
}

# Names country_converter cannot resolve safely on its own — mapped by hand.
# See tests/test_cleaning.py::test_manual_iso_overrides for the fixture that
# proves this table is both necessary and sufficient for the current FAOSTAT area list.
MANUAL_ISO_OVERRIDES = {
    "China, mainland": "CHN",
    "China, Taiwan Province of": "TWN",   # would otherwise return ['CHN', 'TWN']
    "China, Hong Kong SAR": "HKG",
    "China, Macao SAR": "MAC",
}


def to_iso3(area_names: list[str], drop: set[str] | None = None) -> dict[str, str | None]:
    """Map FAOSTAT area names to ISO3, applying manual overrides first.

    Defends against two known failure modes, both independent of what the
    caller remembers to pre-filter:
      1. Names in `drop` (default DROP_AREAS) resolve to None even if
         country_converter would confidently — and wrongly — map them to a
         real country's code (e.g. "Southern Africa" -> ZAF, "Micronesia" ->
         FSM). Aggregate names must never silently blend into a real
         country's data.
      2. country_converter echoes the ORIGINAL NAME back when nothing
         matched (e.g. "Western Europe"), which must never be mistaken for
         a valid ISO3 code.

    Returns exactly one entry per name in area_names — never more, never less."""
    drop = DROP_AREAS if drop is None else drop
    cc = coco.CountryConverter()
    todo = [a for a in area_names if a not in MANUAL_ISO_OVERRIDES and a not in drop]
    converted = cc.convert(todo, to="ISO3", not_found=None) if todo else []
    if isinstance(converted, str):
        converted = [converted]
    resolved = dict(zip(todo, converted))
    resolved.update(MANUAL_ISO_OVERRIDES)
    resolved.update({name: None for name in drop})

    def _valid(code):
        return code if isinstance(code, str) and len(code) == 3 and code.isupper() else None

    return {name: _valid(resolved.get(name)) for name in area_names}


def _mid_year(year_label: str) -> int:
    """FAOSTAT labels a 3-year average by its range, e.g. '2020-2022' -> 2021."""
    s = str(year_label)
    return int(s.split("-")[0]) + 1 if "-" in s else int(s)


def extract_faostat_indicators(fao_raw: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """Filter to the 4 exact indicators, apply FAO's reporting conventions,
    and pivot to one row per (Area, year)."""
    fao = fao_raw[fao_raw["Item"].isin(FAO_ITEMS)].copy()
    fao["year"] = fao["Year"].apply(_mid_year)
    fao["Value"] = pd.to_numeric(
        fao["Value"].astype(str).str.replace("<2.5", "2.5", regex=False), errors="coerce")
    fao["ind"] = fao["Item"].map(FAO_ITEMS)
    fao = fao[fao["year"].between(year_min, year_max)]
    return (fao.pivot_table(index=["Area", "year"], columns="ind", values="Value", aggfunc="first")
               .reset_index())


def reshape_worldbank(wb_frames: dict[str, pd.DataFrame], year_min: int, year_max: int) -> pd.DataFrame:
    """Wide (year=column) -> long (year=row), one merged table across all 4 indicators."""
    years = [str(y) for y in range(year_min, year_max + 1)]
    long_df = None
    for name, df in wb_frames.items():
        if name == "country_classification":
            continue
        keep = ["Country Name", "Country Code"] + [y for y in years if y in df.columns]
        t = df[keep].melt(id_vars=["Country Name", "Country Code"], var_name="year", value_name=name)
        t["year"] = t["year"].astype(int)
        t = t.rename(columns={"Country Code": "country_iso", "Country Name": "wb_name"})
        long_df = t if long_df is None else long_df.merge(
            t.drop(columns="wb_name"), on=["country_iso", "year"], how="outer")
    return long_df


def normalize_classification(country_classification: pd.DataFrame) -> pd.DataFrame:
    """World Bank's metadata CSV uses inconsistent casing (Country Code, Region,
    IncomeGroup — no reliable single naive rule covers all three), so columns
    are matched explicitly rather than guessed with a generic lower/replace."""
    meta = country_classification.copy()
    rename = {}
    for c in meta.columns:
        lc = c.lower().replace(" ", "")
        if lc in ("countrycode", "iso3", "code"):
            rename[c] = "country_iso"
        elif lc == "region":
            rename[c] = "region"
        elif lc == "incomegroup":
            rename[c] = "income_group"
    meta = meta.rename(columns=rename)
    missing = {"country_iso", "region", "income_group"} - set(meta.columns)
    assert not missing, f"country_classification is missing expected columns: {missing}"
    return meta[["country_iso", "region", "income_group"]]


def build_master_dataset(fao_raw: pd.DataFrame, wb_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full cleaning & integration pipeline: FAOSTAT + World Bank -> one master panel.
    This is the ONLY function that should be called to produce master_dataset.csv —
    the notebook and any future script call this, never re-implement the steps inline."""
    s = get_settings()["data"]
    year_min, year_max = s["year_min"], s["year_max"]

    fao_wide = extract_faostat_indicators(fao_raw, year_min, year_max)

    keep_areas = fao_wide[~fao_wide["Area"].isin(DROP_AREAS)].copy()
    iso_map = to_iso3(list(keep_areas["Area"].unique()))
    keep_areas["country_iso"] = keep_areas["Area"].map(iso_map)
    keep_areas = keep_areas[keep_areas["country_iso"].notna()]

    dup = keep_areas.duplicated(subset=["country_iso", "year"]).sum()
    assert dup == 0, f"ISO3 mapping produced {dup} duplicate (country, year) rows — check MANUAL_ISO_OVERRIDES"

    wb_long = reshape_worldbank(wb_frames, year_min, year_max)
    meta = normalize_classification(wb_frames["country_classification"])

    merged = (keep_areas.drop(columns=["Area"])
                        .merge(wb_long.drop(columns="wb_name"), on=["country_iso", "year"], how="inner")
                        .merge(meta, on="country_iso", how="left"))
    merged = merged[merged["region"].notna()].copy()   # drops any residual aggregate
    merged["country_name"] = coco.CountryConverter().convert(list(merged["country_iso"]), to="name_short")

    # 50%-missing rule: a row missing half its raw indicators is mostly invented
    # after imputation, so it is dropped rather than filled.
    missing_share = merged[FEATURES_RAW].isna().mean(axis=1)
    kept = merged[missing_share < s["missing_threshold"]].copy()
    kept = kept.sort_values(["country_iso", "year"]).reset_index(drop=True)

    # Small-gap interpolation only: a gap of <= interpolation_max_gap years can be
    # trusted to a straight line; anything longer is left missing and flagged as such.
    for col in FEATURES_RAW:
        kept[col + "_filled"] = 0
        filled = kept.groupby("country_iso")[col].transform(
            lambda series: series.interpolate(limit=s["interpolation_max_gap"], limit_area="inside"))
        kept.loc[kept[col].isna() & filled.notna(), col + "_filled"] = 1
        kept[col] = filled

    master_cols = (["country_iso", "country_name", "year", "region", "income_group"]
                   + FEATURES_RAW + [c + "_filled" for c in FEATURES_RAW])
    master = kept[master_cols].copy()

    # Final asserts — fail loudly here, not three steps downstream.
    assert master.duplicated(subset=["country_iso", "year"]).sum() == 0
    assert master["year"].between(year_min, year_max).all()
    assert master["pou"].dropna().between(0, 100).all()
    assert master["country_iso"].str.len().eq(3).all()

    return master
