from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("orders_dataset.csv")
TARGET_COLUMN = "returned"

CATEGORICAL_FEATURES = ["product_category", "payment_method"]
NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]


def build_preprocessing_pipeline() -> Pipeline:
    """Build an unfitted preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocessor", transformer)])


def preprocess_data(data_path: Path = DATA_PATH):
    data = pd.read_csv(data_path)

    # Keep identifiers and the target out of the model inputs.
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET_COLUMN]

    # Split raw data first. The test split has no influence on fitted statistics.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    return pipeline, X_train_transformed, X_test_transformed, y_train, y_test


if __name__ == "__main__":
    _, X_train, X_test, y_train, y_test = preprocess_data()
    print(f"Training split: X={X_train.shape}, y={y_train.shape}")
    print(f"Test split: X={X_test.shape}, y={y_test.shape}")
    print(
        "Missing values after preprocessing: "
        f"train={int(X_train.isna().sum().sum())}, "
        f"test={int(X_test.isna().sum().sum())}"
    )
