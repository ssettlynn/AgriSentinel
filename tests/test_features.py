"""
Regression test: a boolean comparison against a missing value (NaN < 5)
silently evaluates to False, not NaN — so a trailing .dropna() on already-
built boolean columns does NOT remove incomplete rows. This test proves
build_transactions() drops incomplete rows BEFORE building the booleans,
by injecting a row with a missing `pou` and confirming it never appears.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agrisentinel import features


def _toy_master(n_years=6):
    years = list(range(2010, 2010 + n_years))
    rows = []
    for i, iso in enumerate(["AAA", "BBB", "CCC"]):
        for y in years:
            rows.append({
                "country_iso": iso, "country_name": iso, "year": y,
                "region": "Test Region", "income_group": "Middle income",
                "pou": 3.0 + i + (y - 2010) * 0.5,
                "des_adequacy": 110.0, "cereal_import_dep": 20.0,
                "food_prod_var": 30.0, "gdp_per_capita": 3000.0 + i * 1000,
                "gdp_growth": 2.0, "inflation_cpi": 4.0, "pop_growth": 1.5,
            })
    return pd.DataFrame(rows)


def test_build_transactions_drops_rows_with_missing_pou_entirely():
    m = _toy_master()
    m = features.add_engineered_features(m)
    m = features.add_targets(m)
    # inject one incomplete row — this is exactly the shape of row that
    # broke mutual exclusivity before the fix
    incomplete_idx = m.index[0]
    m.loc[incomplete_idx, "pou"] = np.nan
    incomplete_key = (m.loc[incomplete_idx, "country_iso"], m.loc[incomplete_idx, "year"])

    tx = features.build_transactions(m)

    still_present = ((tx["country_iso"] == incomplete_key[0]) & (tx["year"] == incomplete_key[1])).any()
    assert not still_present, "row with missing pou must not survive into the transactions table"


def test_build_transactions_hunger_bands_are_mutually_exclusive():
    m = _toy_master()
    m = features.add_engineered_features(m)
    m = features.add_targets(m)
    tx = features.build_transactions(m)
    assert (tx[["HUNGER_LOW", "HUNGER_MED", "HUNGER_HIGH"]].sum(axis=1) == 1).all()
    assert (tx[["INCOME_LOW", "INCOME_MID", "INCOME_HIGH"]].sum(axis=1) == 1).all()


def test_lagged_views_have_all_three_classes_per_split():
    m = _toy_master(n_years=12)
    m = features.add_engineered_features(m)
    m = features.add_targets(m)
    # with only 3 toy countries this won't have 3 classes — so just check the
    # function raises the expected AssertionError rather than silently passing
    try:
        features.build_lagged_view(m, "1yr")
        raised = False
    except AssertionError:
        raised = True
    assert raised, "toy fixture has only one tier — build_lagged_view should refuse it, not silently pass"
