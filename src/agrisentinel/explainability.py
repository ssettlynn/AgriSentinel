"""
src/agrisentinel/explainability.py — the two "why" tools: SHAP for the
individual models, Apriori for cross-cutting patterns the models alone
don't surface (this is what explains the import-dependency null finding
from the descriptive EDA — see docs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.pipeline import Pipeline

from src.agrisentinel.features import FEAT10


def global_shap_importance(fitted_rf: Pipeline, df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Mean |SHAP value| per feature on the test split, for one horizon's RF."""
    test = df[df["split"] == "test"]
    imp, clf = fitted_rf.named_steps["imp"], fitted_rf.named_steps["clf"]
    x = pd.DataFrame(imp.transform(test[FEAT10]), columns=FEAT10)
    shap_values = np.array(shap.TreeExplainer(clf).shap_values(x))
    # shap_values shape can be (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
    feat_axis = [i for i in range(shap_values.ndim) if shap_values.shape[i] == len(FEAT10)][0]
    other_axes = tuple(i for i in range(shap_values.ndim) if i != feat_axis)
    mean_abs = np.abs(shap_values).mean(axis=other_axes)
    ranking = dict(sorted(zip(FEAT10, mean_abs.tolist()), key=lambda kv: -kv[1]))
    return {k: round(v, 4) for k, v in ranking.items()}


def local_shap_by_country(fitted_rf: Pipeline, demo_latest: pd.DataFrame,
                           feats: list[str] | None = None) -> dict[str, dict]:
    """Per-country SHAP: why did THIS country get THIS tier?

    global_shap_importance() answers "what drives the model on average" — it is
    a mean of |SHAP| over the whole test set, so every country shares one
    ranking. That is the wrong object for a country page, where the question is
    "why was Afghanistan called High?".

    This returns SIGNED contributions toward each country's own predicted class,
    so the sign carries meaning that the global mean throws away:
        positive -> this feature pushed the country TOWARD its predicted tier
        negative -> this feature pushed AWAY from it (the model predicted the
                    tier despite this feature, not because of it)

    Computed on `demo_latest` (the latest complete year, targets empty) because
    that is the row the published forecast is actually made from — the same row
    build_predictions() scores. Explaining any other row would explain a
    prediction the site does not show.

    Returns {iso: {"tier": <predicted tier>, "contrib": {feature: signed value}}},
    contributions ordered by descending |value| so a UI can take the top N.
    """
    feats = feats or FEAT10
    imp, clf = fitted_rf.named_steps["imp"], fitted_rf.named_steps["clf"]
    x = pd.DataFrame(imp.transform(demo_latest[feats]), columns=feats)

    classes = list(clf.classes_)
    pred = clf.predict(x)
    sv = np.array(shap.TreeExplainer(clf).shap_values(x))

    # Normalize to (n_rows, n_features, n_classes). SHAP returns either
    # (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
    # depending on version — global_shap_importance() tolerates both, so
    # this must too rather than assuming one shape.
    if sv.ndim == 3 and sv.shape[0] == len(classes) and sv.shape[1] == len(x):
        sv = np.transpose(sv, (1, 2, 0))
    if sv.ndim != 3 or sv.shape[:2] != (len(x), len(feats)):
        raise ValueError(f"unexpected SHAP shape {sv.shape} for "
                          f"{len(x)} rows x {len(feats)} features x {len(classes)} classes")

    out: dict[str, dict] = {}
    for i, iso in enumerate(demo_latest["country_iso"].values):
        cls_idx = classes.index(pred[i])
        contrib = sorted(zip(feats, sv[i, :, cls_idx].tolist()),
                          key=lambda kv: -abs(kv[1]))
        out[iso] = {"tier": str(pred[i]),
                    "contrib": {k: round(float(v), 4) for k, v in contrib}}
    return out


def mine_association_rules(transactions: pd.DataFrame, min_support: float = 0.05,
                            min_lift: float = 1.2) -> pd.DataFrame:
    """Apriori over the binary item columns. Returns rules sorted by lift,
    with antecedent/consequent as readable '+'-joined item names."""
    item_cols = [c for c in transactions.columns if c not in ("country_iso", "year")]
    basket = transactions[item_cols].astype(bool)

    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)

    def _fmt(frozenset_items):
        return " + ".join(sorted(frozenset_items))

    out = pd.DataFrame({
        "antecedent": rules["antecedents"].apply(_fmt),
        "consequent": rules["consequents"].apply(_fmt),
        "support": rules["support"].round(3),
        "confidence": rules["confidence"].round(3),
        "lift": rules["lift"].round(3),
    })
    # Sort by lift, then break ties on the rule text itself (not left to whatever
    # order mlxtend/pandas happen to return) — otherwise two runs on identical
    # data can emit the SAME 52 rules in a different order, which looks like a
    # reproducibility bug in a diff even though nothing about the analysis changed.
    return out.sort_values(["lift", "antecedent", "consequent"],
                            ascending=[False, True, True]).reset_index(drop=True)


def permutation_sanity_check(fitted_rf, df: pd.DataFrame, target_col: str,
                              n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Do two independent importance methods (SHAP and permutation importance)
    agree on the ranking? If the top features agree, that is reassurance the
    SHAP ranking is not an artifact of one particular method."""
    from sklearn.inspection import permutation_importance
    from src.agrisentinel.features import FEAT10

    test = df[df["split"] == "test"]
    result = permutation_importance(fitted_rf, test[FEAT10], test[target_col],
                                     n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    ranking = pd.DataFrame({"feature": FEAT10, "perm_importance_mean": result.importances_mean.round(4),
                             "perm_importance_std": result.importances_std.round(4)})
    return ranking.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
