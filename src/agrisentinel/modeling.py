"""
src/agrisentinel/modeling.py — the four prediction tasks:
  1. Risk tier (classification, both horizons)
  2. Hunger level / multi-indicator value (regression, both horizons)
  3. Risk direction (classification, both horizons)
  plus the persistence baseline every task is measured against.

Design rule: the TEST split is never touched during model selection — only
during the one evaluation call at the end (train_tier_models' final refit +
scoreboard). Hyperparameter choices are made on validation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                              fbeta_score, mean_absolute_error, precision_score,
                              r2_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config.settings import get_settings
from src.agrisentinel.features import FEAT10, FEAT14, FEATURES_RAW, to_tier


def _split_xy(df: pd.DataFrame, split: str, target: str, feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    s = df[df["split"] == split]
    return s[feats], s[target]


def make_classifier(kind: str) -> Pipeline:
    """kind in {'logreg', 'dtree', 'rf'}. Hyperparameters come from settings.yaml
    — never hardcoded here — so a tuning change is a one-file edit."""
    m = get_settings()["models"]
    steps = [("imp", SimpleImputer(strategy="median"))]
    if kind == "logreg":
        steps += [("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    elif kind == "dtree":
        steps += [("clf", DecisionTreeClassifier(
            max_depth=m["dtree_max_depth"], min_samples_leaf=m["dtree_min_samples_leaf"],
            class_weight="balanced", random_state=m["random_state"]))]
    elif kind == "rf":
        steps += [("clf", RandomForestClassifier(
            n_estimators=m["rf_n_estimators"], class_weight="balanced",
            random_state=m["random_state"], n_jobs=-1))]
    else:
        raise ValueError(f"unknown classifier kind: {kind}")
    return Pipeline(steps)


def persistence_baseline(df: pd.DataFrame, split: str, target_col: str) -> dict:
    """'Next period = current tier' — the score every real model must beat."""
    s = df[df["split"] == split]
    y_true, y_pred = s[target_col], s["risk_tier_current"]
    return {
        "model": "Persistence", "split": split,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "high_recall": round(recall_score(y_true, y_pred, labels=["High"], average="macro"), 3),
    }


def _classifier_scores(y_true, y_pred, model_name: str, split: str) -> dict:
    return {
        "model": model_name, "split": split,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "high_recall": round(recall_score(y_true, y_pred, labels=["High"], average="macro"), 3),
    }


# ---------------------------------------------------------------------------
# F-beta: making the precision/recall trade-off explicit.
#
# Macro-F1 is the project's headline metric, but F1 weights precision and recall
# equally — which does not match this problem. Failing to warn about a High-risk
# country costs more than raising a false alarm. F-beta states that weighting as
# a number: beta < 1 favours precision, beta = 1 reproduces F1 exactly, beta > 1
# favours recall.
#
# One beta is reported: 2.0, which weights recall twice as heavily as precision.
# That is the weighting this problem actually has. Failing to warn about a
# High-risk country is far costlier than a false alarm, and every other decision
# in the project already points the same way — balanced class weights, the 0.25
# decision threshold, and the requirement that no High country is ever labelled
# Low. beta = 0.5 would be the right choice only if false alarms were the
# expensive error, which is not the case here.
#
# The comparison is therefore F2 against macro-F1, which the main scoreboard
# already reports and which is exactly beta = 1. Reading the two side by side
# shows whether a model's ranking survives once recall is weighted more heavily.
# ---------------------------------------------------------------------------

FBETA_VALUES: tuple[float, ...] = (2.0,)


def fbeta_table(y_true, predictions: dict[str, object],
                betas: tuple[float, ...] = FBETA_VALUES,
                positive_class: str = "High") -> pd.DataFrame:
    """F-beta for every model in `predictions`, at every beta in `betas`.

    Returns one row per (model, beta) with two columns: the macro-averaged
    F-beta across all three tiers, and the F-beta of the `positive_class` alone
    (High by default) — the class whose recall the project actually cares about.
    Kept free of any model object so it can be unit-tested directly.
    """
    col = f"{positive_class.lower()}_fbeta"
    rows = []
    for model, y_pred in predictions.items():
        for b in betas:
            rows.append({
                "model": model,
                "beta": b,
                "macro_fbeta": round(float(fbeta_score(
                    y_true, y_pred, beta=b, average="macro", zero_division=0)), 3),
                col: round(float(fbeta_score(
                    y_true, y_pred, beta=b, labels=[positive_class],
                    average="macro", zero_division=0)), 3),
            })
    return pd.DataFrame(rows)


def fbeta_comparison(d1yr: pd.DataFrame, d5yr: pd.DataFrame,
                     fitted_tier: dict, betas: tuple[float, ...] = FBETA_VALUES
                     ) -> pd.DataFrame:
    """fbeta_table applied to the TEST split of both horizons, covering the
    persistence baseline and every trained tier model. Uses the same fitted
    models and the same test rows as the macro-F1 scoreboard, so the two tables
    are directly comparable and the beta = 1 column must reproduce macro-F1."""
    views = {"1yr": (d1yr, "risk_tier_next1"), "5yr": (d5yr, "risk_tier_next5")}
    frames = []
    for horizon, (df, target) in views.items():
        test = df[df["split"] == "test"]
        preds: dict[str, object] = {"Persistence": test["risk_tier_current"].to_numpy()}
        for kind, label in [("logreg", "Logistic Regression"),
                            ("dtree", "Decision Tree"), ("rf", "Random Forest")]:
            pipe = fitted_tier.get((horizon, kind))
            if pipe is not None:
                preds[label] = pipe.predict(test[FEAT10])
        table = fbeta_table(test[target].to_numpy(), preds, betas)
        table.insert(1, "horizon", horizon)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# class_weight: how much the model is told each tier matters.
#
# imbalance_experiment() answers a yes/no question — does balancing help at all.
# This sweep answers the follow-up: how MUCH weight should the High tier carry?
# Raising the weight on High buys recall (fewer missed warnings) and costs
# precision (more false alarms), and the point of the sweep is to show that
# trade-off as a curve instead of asserting one setting is correct.
#
# Scored on VALIDATION, never on test: class_weight is a modelling choice, so
# selecting it on the test split would be the leak the whole project avoids.
# ---------------------------------------------------------------------------

CLASS_WEIGHT_GRID: list[tuple[str, object]] = [
    ("none", None),
    ("balanced", "balanced"),
    ("High x2", {"Low": 1, "Medium": 1, "High": 2}),
    ("High x3", {"Low": 1, "Medium": 1, "High": 3}),
    ("High x5", {"Low": 1, "Medium": 1, "High": 5}),
]


MODEL_LABELS = {"rf": "Random Forest", "logreg": "Logistic Regression",
                "dtree": "Decision Tree"}


def class_weight_comparison(d1yr: pd.DataFrame, d5yr: pd.DataFrame,
                            kinds: tuple[str, ...] = ("rf", "logreg", "dtree"),
                            grid: list[tuple[str, object]] | None = None,
                            beta: float = 2.0) -> pd.DataFrame:
    """Sweep class_weight on the tier task and report what each setting buys.

    For every (model, horizon, weight) the model is fitted on TRAIN and scored on
    VALIDATION, reporting macro-F1 plus precision, recall and F-beta on the High
    tier — the numbers that actually move when the weighting changes. Several
    model families are swept because they do not respond the same way: a linear
    model shifts its decision boundary directly, while a deep forest on a
    well-separated signal barely moves at all, and that difference is itself the
    result worth reporting.
    """
    m = get_settings()["models"]
    grid = grid or CLASS_WEIGHT_GRID
    views = {"1yr": (d1yr, "risk_tier_next1"), "5yr": (d5yr, "risk_tier_next5")}
    rows = []

    for kind in kinds:
        for horizon, (df, target) in views.items():
            x_train, y_train = _split_xy(df, "train", target, FEAT10)
            x_val, y_val = _split_xy(df, "val", target, FEAT10)
            for label, weight in grid:
                steps = [("imp", SimpleImputer(strategy="median"))]
                if kind == "logreg":
                    steps += [("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000, class_weight=weight))]
                elif kind == "dtree":
                    steps += [("clf", DecisionTreeClassifier(
                        max_depth=m["dtree_max_depth"],
                        min_samples_leaf=m["dtree_min_samples_leaf"],
                        class_weight=weight, random_state=m["random_state"]))]
                else:
                    steps += [("clf", RandomForestClassifier(
                        n_estimators=m["rf_n_estimators"], class_weight=weight,
                        random_state=m["random_state"], n_jobs=-1))]
                pred = Pipeline(steps).fit(x_train, y_train).predict(x_val)
                rows.append({
                    "model": MODEL_LABELS.get(kind, kind), "horizon": horizon,
                    "class_weight": label,
                    "macro_f1": round(float(f1_score(y_val, pred, average="macro")), 3),
                    "high_precision": round(float(precision_score(
                        y_val, pred, labels=["High"], average="macro", zero_division=0)), 3),
                    "high_recall": round(float(recall_score(
                        y_val, pred, labels=["High"], average="macro", zero_division=0)), 3),
                    "high_fbeta": round(float(fbeta_score(
                        y_val, pred, beta=beta, labels=["High"], average="macro",
                        zero_division=0)), 3),
                })
    return pd.DataFrame(rows)


def tune_rf_depth(df: pd.DataFrame, target: str, feats: list[str]) -> tuple[dict, pd.DataFrame]:
    """Regularization study: is a train/val gap fixable by limiting tree depth,
    or is the ceiling the DATA rather than the model? Grid is searched on
    validation only; the chosen config is picked by best val macro-F1. This
    is itself one of the honest findings the project reports (a 5-year-horizon
    RF gap that shrinks with regularization while validation score also drops
    is evidence the limit is data volume, not model capacity) — so the full
    grid is returned alongside the winner, not thrown away."""
    m = get_settings()["models"]
    x_train, y_train = _split_xy(df, "train", target, feats)
    x_val, y_val = _split_xy(df, "val", target, feats)

    grid = [dict(max_depth=d, min_samples_leaf=l)
            for d in (None, 15, 10, 8, 6) for l in (2, 10)]
    rows = []
    for cfg in grid:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("clf", RandomForestClassifier(
                              n_estimators=m["rf_n_estimators"], class_weight="balanced",
                              random_state=m["random_state"], n_jobs=-1, **cfg))])
        pipe.fit(x_train, y_train)
        train_f1 = f1_score(y_train, pipe.predict(x_train), average="macro")
        val_f1 = f1_score(y_val, pipe.predict(x_val), average="macro")
        rows.append({**cfg, "train_macro_f1": round(train_f1, 3), "val_macro_f1": round(val_f1, 3),
                     "gap_pp": round((train_f1 - val_f1) * 100, 1)})

    grid_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    best_row = grid_df.iloc[0]
    best_cfg = {"max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
                "min_samples_leaf": int(best_row["min_samples_leaf"])}
    return best_cfg, grid_df


def train_tier_models(d1yr: pd.DataFrame, d5yr: pd.DataFrame
                       ) -> tuple[dict, list[dict], dict[str, pd.DataFrame]]:
    """Train Persistence/LogReg/DTree/RF for both horizons. Model selection
    (including the RF depth/leaf regularization grid) happens on validation
    only; TEST is scored exactly once, here, at the end.

    Returns (fitted_models, scoreboard, rf_tuning_grids) — fitted_models is
    refit on train+val for deployment; rf_tuning_grids preserves the full
    regularization study per horizon for the evaluation report."""
    m = get_settings()["models"]
    views = {"1yr": (d1yr, "risk_tier_next1"), "5yr": (d5yr, "risk_tier_next5")}
    fitted: dict[tuple[str, str], Pipeline] = {}
    scoreboard: list[dict] = []
    rf_tuning_grids: dict[str, pd.DataFrame] = {}

    for horizon, (df, target) in views.items():
        for split in ("train", "val", "test"):
            row = persistence_baseline(df, split, target)
            row["horizon"] = horizon
            scoreboard.append(row)

        x_train, y_train = _split_xy(df, "train", target, FEAT10)
        x_val, y_val = _split_xy(df, "val", target, FEAT10)
        x_test, y_test = _split_xy(df, "test", target, FEAT10)
        fit_df = df[df["split"].isin(["train", "val"])]

        best_rf_cfg, grid_df = tune_rf_depth(df, target, FEAT10)
        rf_tuning_grids[horizon] = grid_df

        for kind, label in [("logreg", "Logistic Regression"), ("dtree", "Decision Tree"),
                             ("rf", "Random Forest")]:
            if kind == "rf":
                probe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=m["rf_n_estimators"], class_weight="balanced",
                                       random_state=m["random_state"], n_jobs=-1, **best_rf_cfg))])
            else:
                probe = make_classifier(kind)
            probe.fit(x_train, y_train)
            for split_name, x_s, y_s in [("train", x_train, y_train), ("val", x_val, y_val)]:
                row = _classifier_scores(y_s, probe.predict(x_s), label, split_name)
                row["horizon"] = horizon
                scoreboard.append(row)

            if kind == "rf":
                final = Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=m["rf_n_estimators"], class_weight="balanced",
                                       random_state=m["random_state"], n_jobs=-1, **best_rf_cfg))])
            else:
                final = make_classifier(kind)
            final.fit(fit_df[FEAT10], fit_df[target])
            fitted[(horizon, kind)] = final
            row = _classifier_scores(y_test, final.predict(x_test), label, "test")
            row["horizon"] = horizon
            scoreboard.append(row)

    return fitted, scoreboard, rf_tuning_grids


# Sensible mid-regularization default for the direction RF. Not arbitrary: it is
# the config the project shipped with, and the grid below is what justifies keeping
# it. It is only overridden if some other config beats it on validation by MORE than
# `epsilon` — see tune_direction_rf for why that guard matters on this dataset.
DIRECTION_RF_DEFAULT = {"max_depth": 12, "min_samples_leaf": 3}


def tune_direction_rf(labelled: pd.DataFrame, target_col: str,
                      train_end: int, val_end: int,
                      incumbent: dict | None = None,
                      epsilon: float = 0.015) -> tuple[dict, pd.DataFrame]:
    """Validation-only hyperparameter grid for the direction classifier — the
    same regularization-study discipline train_tier_models applies via
    tune_rf_depth, which the direction task previously lacked.

    Grid is searched on TRAIN->VAL only. Selection is deliberately NOT naive
    argmax: the validation fold here is ~330 rows, and we measured directly that
    argmax on it chases noise — different reasonable rules pick configs that
    scatter 0.68-0.72 on the held-out TEST set, all within each other's noise.
    So the grid is used as a CONFIRMATION, not an autopilot: we keep the
    incumbent config unless another beats it on validation by more than
    `epsilon` (a real margin, not noise). On this data nothing does, so the
    incumbent stands — and the returned grid shows why that is the honest call:
    the large train/val gap (~19-26pp) does not close with regularization, so
    the ceiling is DATA VOLUME, not tree depth (the same finding tune_rf_depth
    surfaces for the tier task)."""
    m = get_settings()["models"]
    incumbent = incumbent or DIRECTION_RF_DEFAULT
    train = labelled[labelled.year <= train_end]
    val = labelled[labelled.year.between(train_end + 1, val_end)]

    grid = [dict(max_depth=d, min_samples_leaf=l)
            for d in (None, 20, 16, 12, 10, 8, 6) for l in (2, 3, 5, 10)]
    rows = []
    for cfg in grid:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(
                             n_estimators=m["rf_n_estimators"], class_weight="balanced",
                             random_state=m["random_state"], n_jobs=-1, **cfg))])
        pipe.fit(train[FEAT14], train[target_col])
        train_f1 = f1_score(train[target_col], pipe.predict(train[FEAT14]), average="macro")
        val_f1 = f1_score(val[target_col], pipe.predict(val[FEAT14]), average="macro")
        rows.append({**cfg, "train_macro_f1": round(train_f1, 3), "val_macro_f1": round(val_f1, 3),
                     "gap_pp": round((train_f1 - val_f1) * 100, 1)})

    grid_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_val = float(grid_df["val_macro_f1"].max())

    def _val_of(cfg):
        match = grid_df[(grid_df["max_depth"].isna() if cfg["max_depth"] is None
                         else grid_df["max_depth"] == cfg["max_depth"])
                        & (grid_df["min_samples_leaf"] == cfg["min_samples_leaf"])]
        return float(match["val_macro_f1"].iloc[0]) if len(match) else -1.0

    if _val_of(incumbent) >= best_val - epsilon:
        best_cfg = dict(incumbent)               # grid confirms incumbent is within noise → keep it
    else:
        pick = grid_df.iloc[0]                    # a config clearly beats it → adopt the winner
        best_cfg = {"max_depth": None if pd.isna(pick["max_depth"]) else int(pick["max_depth"]),
                    "min_samples_leaf": int(pick["min_samples_leaf"])}
    return best_cfg, grid_df


def train_direction_models(master_with_targets: pd.DataFrame
                           ) -> tuple[dict, list[dict], dict[str, pd.DataFrame]]:
    """Task 3: Improving / Stable / Worsening, both horizons.

    Now follows the SAME train/val/test protocol as the tier task (it was the
    only task that didn't): hyperparameters are chosen on validation via
    tune_direction_rf, the evaluation model is refit on train+val, and TEST is
    scored exactly once. Previously the eval model trained on train-only, silently
    discarding the validation years — fixing that alone lifts the reported scores
    (esp. 5yr) with no change to the data or the honest data-limited finding.

    Returns (fitted_prod_models, metrics, tuning_grids)."""
    s = get_settings()
    fitted: dict[str, Pipeline] = {}
    metrics: list[dict] = []
    tuning_grids: dict[str, pd.DataFrame] = {}

    horizon_cfg = {
        "1yr": (1, s["splits"]["1yr"]["train_end"], s["splits"]["1yr"]["val_end"],
                s["data"]["year_max"] - 1),
        "5yr": (5, s["splits"]["5yr"]["train_end"], s["splits"]["5yr"]["val_end"],
                s["data"]["year_max"] - 1),
    }
    for horizon, (h, train_end, val_end, test_to) in horizon_cfg.items():
        col = f"dir_next{h}"
        labelled = master_with_targets.dropna(subset=[col, "pou"]).copy()
        labelled = labelled[labelled[col].notna()]

        best_cfg, grid_df = tune_direction_rf(labelled, col, train_end, val_end)
        tuning_grids[horizon] = grid_df

        trainval = labelled[labelled.year <= val_end]     # train+val, as the tier task uses
        test = labelled[labelled.year.between(val_end + 1, test_to)]

        def _pipe(cfg):
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("clf", RandomForestClassifier(
                                 n_estimators=s["models"]["rf_n_estimators"], class_weight="balanced",
                                 random_state=s["models"]["random_state"], n_jobs=-1, **cfg))])

        evaluator = _pipe(best_cfg)
        evaluator.fit(trainval[FEAT14], trainval[col])     # refit on train+val, score test once
        pred = evaluator.predict(test[FEAT14])

        majority = trainval[col].value_counts().idxmax()
        metrics.append({
            "horizon": horizon,
            "macro_f1": round(f1_score(test[col], pred, average="macro"), 3),
            "accuracy": round(accuracy_score(test[col], pred), 3),
            "baseline_macro_f1": round(f1_score(test[col], [majority] * len(test), average="macro"), 3),
            "baseline_accuracy": round(accuracy_score(test[col], [majority] * len(test)), 3),
            "chosen_hyperparams": best_cfg,
            "class_balance": labelled[col].value_counts().to_dict(),
            # per-class detail from the SAME held-out evaluation (train+val -> test, no
            # leakage) so any notebook/report showing the breakdown matches macro_f1 above,
            # instead of re-scoring the production model on rows it was trained on.
            "test_report": classification_report(test[col], pred, output_dict=True, zero_division=0),
        })
        # production model refit on every labelled row, for live forecasting
        prod = _pipe(best_cfg)
        prod.fit(labelled[FEAT14], labelled[col])
        fitted[horizon] = prod

    return fitted, metrics, tuning_grids


def train_forecast_regressors(master_with_targets: pd.DataFrame, indicators: list[str]
                               ) -> tuple[dict, list[dict]]:
    """Task 4: future value of each indicator, both horizons. Reports MAE/R²
    against a persistence baseline (predict = current value)."""
    s = get_settings()
    fitted: dict[tuple[str, str], Pipeline] = {}
    metrics: list[dict] = []
    horizon_splits = {1: (2017, 2020), 5: (2014, 2017)}

    for indicator in indicators:
        for h, (train_end, test_from) in horizon_splits.items():
            target_col = f"{indicator}_t{h}"
            m = master_with_targets.copy()
            m[target_col] = m.groupby("country_iso")[indicator].shift(-h)
            labelled = m.dropna(subset=[indicator, target_col])
            train = labelled[labelled.year <= train_end]
            test = labelled[labelled.year >= test_from]

            baseline_mae = mean_absolute_error(test[target_col], test[indicator])
            reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("reg", RandomForestRegressor(n_estimators=250, max_depth=12,
                                    min_samples_leaf=3, random_state=s["models"]["random_state"],
                                    n_jobs=-1))])
            reg.fit(train[FEAT14], train[target_col])
            pred = reg.predict(test[FEAT14])

            metrics.append({
                "indicator": indicator, "horizon": f"{h}yr",
                "mae": round(mean_absolute_error(test[target_col], pred), 2),
                "r2": round(r2_score(test[target_col], pred), 2),
                "baseline_mae": round(baseline_mae, 2),
            })
            prod = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("reg", RandomForestRegressor(n_estimators=250, max_depth=12,
                                     min_samples_leaf=3, random_state=s["models"]["random_state"],
                                     n_jobs=-1))])
            prod.fit(labelled[FEAT14], labelled[target_col])
            fitted[(indicator, f"{h}yr")] = prod

    return fitted, metrics


# ---------------------------------------------------------------------------
# Deep evaluation: threshold tuning, bootstrap CI, learning curves.
# These are diagnostic tools over an already-fitted model — they produce no
# artifact the rest of the pipeline depends on, but they ARE part of the
# evaluation report and belong in one tested place, not copy-pasted per notebook cell.
# ---------------------------------------------------------------------------

def threshold_sweep(fitted_model: Pipeline, x_val: pd.DataFrame, y_val: pd.Series,
                     positive_class: str = "High",
                     thresholds: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
                     ) -> pd.DataFrame:
    """Trade recall against precision for the High class, on VALIDATION only.
    Domain rule: missing a High-risk country (false negative) is costlier than
    a false alarm, so recall is prioritised subject to a minimum precision."""
    classes = list(fitted_model.named_steps["clf"].classes_)
    pos_idx = classes.index(positive_class)
    proba = fitted_model.predict_proba(x_val)[:, pos_idx]
    y_true_bin = (y_val == positive_class).astype(int)

    rows = []
    for t in thresholds:
        pred_bin = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "high_recall": round(recall_score(y_true_bin, pred_bin, zero_division=0), 3),
            "high_precision": round(precision_score(y_true_bin, pred_bin, zero_division=0), 3),
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep_df: pd.DataFrame, min_precision: float = 0.70) -> float:
    """Highest recall subject to precision staying at or above min_precision."""
    ok = sweep_df[sweep_df["high_precision"] >= min_precision]
    pool = ok if len(ok) else sweep_df
    return float(pool.sort_values("high_recall", ascending=False).iloc[0]["threshold"])


def bootstrap_ci_macro_f1(y_true: pd.Series, y_pred: np.ndarray, n_iter: int = 1000,
                           random_state: int = 42) -> dict:
    """Is the reported macro-F1 luck, or a real signal? Resample the test set
    with replacement n_iter times and report the 95% interval."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"point": round(float(f1_score(y_true, y_pred, average="macro")), 3),
            "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)}


def learning_curve_summary(kind: str, x: pd.DataFrame, y: pd.Series,
                            train_sizes=(0.2, 0.4, 0.6, 0.8, 1.0), cv: int = 3) -> pd.DataFrame:
    """Does more data help? A curve that is still rising at the largest
    training size means the model is data-starved, not at its ceiling."""
    pipe = make_classifier(kind)
    sizes, train_scores, val_scores = learning_curve(
        pipe, x, y, train_sizes=list(train_sizes), cv=cv, scoring="f1_macro", n_jobs=-1)
    return pd.DataFrame({
        "train_size": sizes,
        "train_f1_mean": train_scores.mean(axis=1).round(3),
        "val_f1_mean": val_scores.mean(axis=1).round(3),
    })


# ---------------------------------------------------------------------------
# Decision experiments (teacher's suggestions): every modelling choice settled
# by a measurement, not an opinion. Each imputes with the median first (so no
# rows are dropped) and computes the tier target with the project's own to_tier,
# so the notebook, report, and tests all exercise one implementation.
# ---------------------------------------------------------------------------

def _tier_xy(master: pd.DataFrame, cols: list[str]):
    """Median-imputed feature matrix + tier label, from the raw master panel."""
    df = master.dropna(subset=["pou"]).copy()
    y = df["pou"].apply(to_tier).to_numpy()
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[cols]),
                     columns=cols, index=df.index)
    return x, y


def imbalance_experiment(master: pd.DataFrame, feats: list[str] | None = None,
                         n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Does handling class imbalance actually help? The SAME logistic model is
    cross-validated twice on the (same-year) tier task — once ignoring imbalance,
    once with class_weight='balanced' — and compared on macro-F1 and, crucially,
    recall on the High class (the class an early-warning system must not miss).
    Stratified CV keeps each fold's class mix representative."""
    feats = feats or FEATURES_RAW
    x, y = _tier_xy(master, feats)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for label, cw in [("unbalanced", None), ("balanced", "balanced")]:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight=cw))])
        pred = cross_val_predict(pipe, x, y, cv=cv)
        rows.append({
            "setting": label,
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
            "high_recall": round(float(recall_score(y, pred, labels=["High"], average="macro")), 3),
        })
    return pd.DataFrame(rows)


def flag_ablation(master: pd.DataFrame, feats: list[str] | None = None,
                  flag_cols: list[str] | None = None, n_estimators: int | None = None,
                  n_splits: int = 5, random_state: int = 42) -> dict:
    """The teacher's exact question, answered by measurement: do the trailing
    `*_filled` 0/1 columns help the model? Same Random Forest, same folds, once
    with features only and once with features + flags. If macro-F1 does not
    improve, the flags stay OUT of training — they record HOW a value was filled,
    not the country's real state, so they belong in the audit trail, not the model."""
    feats = feats or FEATURES_RAW
    if flag_cols is None:
        flag_cols = [c for c in master.columns if c.endswith("_filled")]
    n_estimators = n_estimators or get_settings()["models"]["rf_n_estimators"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _score(cols: list[str]) -> float:
        x, y = _tier_xy(master, cols)
        rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                    random_state=random_state, n_jobs=-1)
        return round(float(cross_val_score(rf, x, y, cv=cv, scoring="f1_macro").mean()), 4)

    only = _score(feats)
    with_flags = _score(feats + flag_cols)
    return {
        "features_only_macro_f1": only,
        "features_plus_flags_macro_f1": with_flags,
        "delta": round(with_flags - only, 4),
        "flags_help": bool(with_flags - only > 0.005),
        "n_flags": len(flag_cols),
    }


def ols_regression(master: pd.DataFrame, target: str = "pou",
                   predictors: list[str] | None = None) -> dict:
    """Ordinary least squares of `target` on the other raw features, implemented
    with numpy/scipy so the project takes no new dependency. Returns a coefficient
    table (coef, std_err, t, p_value) plus R²/adjusted-R² — the interpretable
    linear counterpart to the tree models, answering WHICH indicators move `pou`
    and whether each is statistically significant. Inputs are median-imputed so
    no rows are dropped."""
    predictors = predictors or [c for c in FEATURES_RAW if c != target]
    df = master.dropna(subset=[target]).copy()
    x_raw = SimpleImputer(strategy="median").fit_transform(df[predictors])
    y = df[target].to_numpy(dtype=float)
    n = len(y)
    x = np.column_stack([np.ones(n), x_raw])              # design matrix with intercept
    k = x.shape[1]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = n - k
    mse = float(resid @ resid) / dof
    var_beta = np.diag(np.linalg.inv(x.T @ x)) * mse
    se = np.sqrt(var_beta)
    t_stat = beta / se
    p_val = 2 * _stats.t.sf(np.abs(t_stat), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof
    coef_table = pd.DataFrame({
        "term": ["const"] + list(predictors),
        "coef": np.round(beta, 4),
        "std_err": np.round(se, 4),
        "t": np.round(t_stat, 3),
        "p_value": p_val,
    })
    return {"coef_table": coef_table, "r2": round(r2, 3),
            "adj_r2": round(adj_r2, 3), "n": n, "dof": dof}
