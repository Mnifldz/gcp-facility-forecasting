from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "pyarrow",
        "lightgbm",
        "scikit-learn",
    ],
)
def train_lightgbm(
    training_data: Input[Dataset],
    trained_model: Output[Model],
):
    import pandas as pd
    import lightgbm as lgb
    import joblib

    df = pd.read_parquet(training_data.path)

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date"]
    )

    train = df[
        df["prediction_date"] <= "2025-08-24"
    ].copy()

    features = [
        "facility_id",
        "day_of_week",
        "bookings_today",
        "bookings_lag_1",
        "bookings_lag_7",
        "bookings_lag_14",
        "bookings_rolling_7",
        "bookings_rolling_28",
    ]

    X_train = train[features].copy()
    y_train = train["bookings_next_7_days"]

    X_train["facility_id"] = (
        X_train["facility_id"].astype("category")
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=["facility_id"],
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "categories": list(
                X_train["facility_id"].cat.categories
            ),
        },
        trained_model.path,
    )

    trained_model.metadata["algorithm"] = "LightGBM"
    trained_model.metadata["training_rows"] = len(train)