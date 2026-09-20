"""
Regression test for lesson #4: "China, Taiwan Province of" matches BOTH the
CHN and TWN patterns in country_converter and returns a LIST, which crashes
any later duplicate-check with "unhashable type: 'list'". This test proves
the manual override table catches every known problem name BEFORE it reaches
country_converter, using a fixed fixture — not a live network call.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agrisentinel.cleaning import to_iso3, MANUAL_ISO_OVERRIDES, DROP_AREAS


# The exact fixture that broke the prior project, plus the other known traps.
KNOWN_PROBLEM_NAMES = [
    "China, mainland",
    "China, Taiwan Province of",   # the one that returns a list if unhandled
    "China, Hong Kong SAR",
    "China, Macao SAR",
]

KNOWN_GOOD_NAMES = ["Myanmar", "Kenya", "Türkiye", "Afghanistan"]

KNOWN_UNRESOLVABLE_AGGREGATES = ["Western Europe", "World", "Southern Africa", "Micronesia"]


def test_manual_overrides_cover_every_multi_match_name():
    result = to_iso3(KNOWN_PROBLEM_NAMES)
    for name in KNOWN_PROBLEM_NAMES:
        assert name in MANUAL_ISO_OVERRIDES, f"{name!r} must have an explicit override"
        code = result[name]
        assert isinstance(code, str) and len(code) == 3 and code.isupper(), (
            f"{name!r} resolved to {code!r} — expected a clean ISO3 string, "
            f"never a list (that is exactly what crashed the prior project)")


def test_taiwan_resolves_to_twn_not_a_list():
    result = to_iso3(["China, Taiwan Province of"])
    assert result["China, Taiwan Province of"] == "TWN"
    assert not isinstance(result["China, Taiwan Province of"], list)


def test_known_good_countries_resolve_cleanly():
    result = to_iso3(KNOWN_GOOD_NAMES)
    expected = {"Myanmar": "MMR", "Kenya": "KEN", "Türkiye": "TUR", "Afghanistan": "AFG"}
    assert result == expected


def test_unresolvable_aggregates_return_none_not_the_original_name():
    # country_converter echoes the input string back when nothing matches;
    # that echoed string must never be mistaken for a valid ISO3 code.
    result = to_iso3(KNOWN_UNRESOLVABLE_AGGREGATES)
    for name in KNOWN_UNRESOLVABLE_AGGREGATES:
        assert result[name] is None, f"{name!r} should resolve to None, got {result[name]!r}"


def test_drop_areas_and_manual_overrides_do_not_overlap():
    # A name should never be BOTH dropped as an aggregate AND manually mapped —
    # that would make its treatment depend on code order, which is a footgun.
    assert not (DROP_AREAS & set(MANUAL_ISO_OVERRIDES)), \
        "DROP_AREAS and MANUAL_ISO_OVERRIDES overlap — resolve the ambiguity explicitly"
