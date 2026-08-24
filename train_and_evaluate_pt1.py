"""Train, evaluate, explain, and persist the return-risk models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from main import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
)


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "return_risk_model.pkl"
REPORT_PATH = ROOT / "model_report.json"
LOGISTIC_SWEEP_PATH = ROOT / "logistic_threshold_sweep.csv"
FOREST_SWEEP_PATH = ROOT / "random_forest_threshold_sweep.csv"
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def threshold_sweep(y_true, probabilities) -> tuple[list[dict], dict]:
    rows = [classification_metrics(y_true, probabilities, t) for t in THRESHOLDS]
    # Explicit tie-breaking makes the result deterministic: prefer higher recall.
    best = max(rows, key=lambda row: (row["f1"], row["recall"]))
    return rows, best


def subgroup_metrics(frame, y_true, probabilities, threshold, column) -> list[dict]:
    predictions = (probabilities >= threshold).astype(int)
    evaluated = pd.DataFrame(
        {
            column: frame[column].to_numpy(),
            "actual": y_true.to_numpy(),
            "predicted": predictions,
        }
    )
    rows = []
    for name, group in evaluated.groupby(column, sort=True):
        rows.append(
            {
                column: str(name),
                "rows": int(len(group)),
                "positives": int(group["actual"].sum()),
                "recall": float(
                    recall_score(group["actual"], group["predicted"], zero_division=0)
                ),
                "precision": float(
                    precision_score(group["actual"], group["predicted"], zero_division=0)
                ),
            }
        )
    return rows


def data_summary(data: pd.DataFrame) -> dict:
    category = (
        data.groupby("product_category")[TARGET_COLUMN]
        .agg(rows="size", returns="sum", return_rate="mean")
        .reset_index()
        .to_dict("records")
    )
    payment = (
        data.groupby("payment_method")[TARGET_COLUMN]
        .agg(rows="size", returns="sum", return_rate="mean")
        .reset_index()
        .to_dict("records")
    )
    missing_by_payment = (
        data.assign(rating_missing=data["rating_given"].isna())
        .groupby("payment_method")["rating_missing"]
        .agg(rows="size", missing="sum", missing_rate="mean")
        .reset_index()
        .to_dict("records")
    )
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "return_rate": float(data[TARGET_COLUMN].mean()),
        "rating_missing_rate": float(data["rating_given"].isna().mean()),
        "return_rate_by_product_category": category,
        "return_rate_by_payment_method": payment,
        "rating_missing_by_payment_method": missing_by_payment,
        "missingness_classification": "MAR",
    }


def run() -> dict:
    data = pd.read_csv(DATA_PATH)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[feature_columns]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)
    dummy_metrics = {
        "accuracy": float(accuracy_score(y_test, dummy_predictions)),
        "f1": float(f1_score(y_test, dummy_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, dummy_predictions, zero_division=0)),
    }

    logistic = Pipeline(
        [
            ("preprocessor", clone(build_preprocessing_pipeline().named_steps["preprocessor"])),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    logistic_probabilities = logistic.predict_proba(X_test)[:, 1]
    logistic_default = classification_metrics(y_test, logistic_probabilities, 0.5)
    logistic_sweep, logistic_best = threshold_sweep(y_test, logistic_probabilities)
    pd.DataFrame(logistic_sweep).to_csv(LOGISTIC_SWEEP_PATH, index=False)

    forest = Pipeline(
        [
            ("preprocessor", clone(build_preprocessing_pipeline().named_steps["preprocessor"])),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        forest,
        param_grid={
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [6, 10, None],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    winning_forest = search.best_estimator_
    forest_probabilities = winning_forest.predict_proba(X_test)[:, 1]
    forest_test_auc = roc_auc_score(y_test, forest_probabilities)
    forest_sweep, forest_best = threshold_sweep(y_test, forest_probabilities)
    pd.DataFrame(forest_sweep).to_csv(FOREST_SWEEP_PATH, index=False)

    preprocessor = winning_forest.named_steps["preprocessor"]
    classifier = winning_forest.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    impurity = pd.Series(classifier.feature_importances_, index=feature_names)
    top_five = impurity.sort_values(ascending=False).head(5)

    transformed_test = preprocessor.transform(X_test)
    perm = permutation_importance(
        classifier,
        transformed_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    permutation = pd.Series(perm.importances_mean, index=feature_names)
    importance_comparison = [
        {
            "feature": feature,
            "impurity_importance": float(top_five[feature]),
            "permutation_importance": float(permutation[feature]),
            "impurity_rank": rank,
            "permutation_rank_among_top_five": int(
                permutation[top_five.index].rank(ascending=False, method="min")[feature]
            ),
        }
        for rank, feature in enumerate(top_five.index, start=1)
    ]

    overall_at_rf_threshold = classification_metrics(
        y_test, forest_probabilities, forest_best["threshold"]
    )
    subgroups = {
        "overall": overall_at_rf_threshold,
        "product_category": subgroup_metrics(
            X_test,
            y_test,
            forest_probabilities,
            forest_best["threshold"],
            "product_category",
        ),
        "payment_method": subgroup_metrics(
            X_test,
            y_test,
            forest_probabilities,
            forest_best["threshold"],
            "payment_method",
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winning_forest, MODEL_PATH)
    loaded_model = joblib.load(MODEL_PATH)
    if not np.allclose(
        loaded_model.predict_proba(X_test)[:, 1], forest_probabilities
    ):
        raise RuntimeError("Persisted model predictions differ after reload")

    report = {
        "data": data_summary(data),
        "split": {"train_rows": len(X_train), "test_rows": len(X_test)},
        "dummy": dummy_metrics,
        "logistic_regression": {
            "default_threshold": logistic_default,
            "best_threshold": logistic_best,
            "threshold_sweep": logistic_sweep,
        },
        "random_forest": {
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "test_roc_auc": float(forest_test_auc),
            "best_threshold": forest_best,
            "threshold_sweep": forest_sweep,
            "top_five_importance_comparison": importance_comparison,
        },
        "subgroups_at_rf_best_threshold": subgroups,
        "artifact": {
            "path": str(MODEL_PATH.relative_to(ROOT)),
            "reload_verified": True,
            "classifier": type(loaded_model.named_steps["classifier"]).__name__,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
