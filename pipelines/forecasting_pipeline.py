from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Metrics, Input, Output


PROJECT_ID = "storage-analytics-lab"
REGION = "us-west2"

PIPELINE_ROOT = (
    "gs://storage-analytics-lab-vertex-artifacts/"
    "pipelines/facility-forecasting"
)

BASE_IMAGE = "python:3.11"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery",
        "pandas",
        "db-dtypes",
        "pyarrow",
    ],
)
def load_training_data(
    training_data: Output[Dataset],
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project="storage-analytics-lab")

    query = """
    SELECT *
    FROM `storage-analytics-lab.storage_ml_dev.facility_forecasting_features`
    ORDER BY prediction_date, facility_id
    """

    df = client.query(query).to_dataframe()

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date"]
    )

    # Write data to the artifact path supplied by Vertex AI.
    df.to_parquet(training_data.path, index=False)

    training_data.metadata["row_count"] = len(df)
    training_data.metadata["source_table"] = (
        "storage-analytics-lab.storage_ml_dev."
        "facility_forecasting_features"
    )


@dsl.component(
    base_image=BASE_IMAGE,
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


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
    "google-cloud-aiplatform",
    "google-cloud-storage",
    "pandas",
    "pyarrow",
    "lightgbm",
    "scikit-learn",
    "joblib",
    ],
)
def evaluate_and_register(
    training_data: Input[Dataset],
    trained_model: Input[Model],
    evaluation_metrics: Output[Metrics],
    registered_model_name: Output[Dataset],
):
    import uuid
    from pathlib import Path

    from google.cloud import aiplatform, storage
    from google.api_core.exceptions import NotFound
    from google.cloud import aiplatform_v1

    import joblib
    import pandas as pd

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
    )

    artifact = joblib.load(trained_model.path)

    model = artifact["model"]
    features = artifact["features"]
    categories = artifact["categories"]

    df = pd.read_parquet(training_data.path)

    validation = df[
        (df["prediction_date"] >= "2025-09-01")
        & (df["prediction_date"] <= "2025-10-24")
    ].copy()

    X_val = validation[features].copy()

    X_val["facility_id"] = pd.Categorical(
        X_val["facility_id"],
        categories=categories,
    )

    y_val = validation["bookings_next_7_days"]

    predictions = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, predictions))

    rmse = float(
        mean_squared_error(y_val, predictions) ** 0.5
    )

    wmape = float(
        abs(y_val - predictions).sum()
        / abs(y_val).sum()
    )

    evaluation_metrics.log_metric("mae", mae)
    evaluation_metrics.log_metric("rmse", rmse)
    evaluation_metrics.log_metric("wmape", wmape)

    print(
        f"Validation: MAE={mae:.4f}, "
        f"RMSE={rmse:.4f}, WMAPE={wmape:.4f}",
        flush=True,
    )

    project = "storage-analytics-lab"
    region = "us-west2"

    bucket_name = "storage-analytics-lab-registered-models"
    model_id = "facility-demand-lightgbm"

    parent_model = (
        "projects/555144694235/"
        "locations/us-west2/"
        f"models/{model_id}"
    )

    image_uri = (
        "us-west2-docker.pkg.dev/"
        "storage-analytics-lab/ml-containers/"
        "lightgbm-inference:v1"
    )

    aiplatform.init(
        project=project,
        location=region,
    )

    # Store this run's artifact in a unique directory.
    run_id = uuid.uuid4().hex

    artifact_prefix = (
        f"facility-demand/{run_id}"
    )

    artifact_uri = (
        f"gs://{bucket_name}/{artifact_prefix}"
    )

    client = storage.Client(project=project)

    blob = client.bucket(bucket_name).blob(
        f"{artifact_prefix}/model.joblib"
    )

    blob.upload_from_filename(trained_model.path)

    print(f"Model artifact: {artifact_uri}", flush=True)

    # Determine whether the parent model already exists.

    model_client = aiplatform_v1.ModelServiceClient(
        client_options={
            "api_endpoint": f"{region}-aiplatform.googleapis.com"
        }
    )

    try:
        model_client.get_model(name=parent_model)
        model_exists = True
        print("Existing registered model found.", flush=True)

    except NotFound:
        model_exists = False
        print("No existing model; creating version 1.", flush=True)

    upload_args = {
        "display_name": "facility-demand-lightgbm",
        "artifact_uri": artifact_uri,
        "serving_container_image_uri": image_uri,
        "serving_container_predict_route": "/predict",
        "serving_container_health_route": "/health",
        "serving_container_ports": [8080],
        "sync": True,
    }

    if model_exists:
        # Add a new version without changing the default alias.
        upload_args["parent_model"] = parent_model
        upload_args["is_default_version"] = False
    else:
        # Create the first version with a predictable model ID.
        upload_args["model_id"] = model_id

    model = aiplatform.Model.upload(**upload_args)

    print(
        f"Registered model: {model.resource_name}",
        flush=True,
    )

    print(
        f"Version ID: {model.version_id}",
        flush=True,
    )

    # Save both identifiers for downstream use.
    Path(registered_model_name.path).write_text(
        f"{model.resource_name}@{model.version_id}"
    )

    from google.cloud import aiplatform_v1
    from google.protobuf.struct_pb2 import Struct

    metrics = Struct()
    metrics.update({
        "meanAbsoluteError": mae,
        "rootMeanSquaredError": rmse,
    })

    evaluation = aiplatform_v1.ModelEvaluation(
        display_name="validation",
        metrics_schema_uri=(
            "gs://google-cloud-aiplatform/schema/"
            "modelevaluation/regression_metrics_1.0.0.yaml"
        ),
        metrics=metrics,
    )

    model_client = aiplatform_v1.ModelServiceClient(
        client_options={
            "api_endpoint": f"{region}-aiplatform.googleapis.com"
        }
    )


    versioned_model_name = (
        f"{model.resource_name}@{model.version_id}"
    )

    imported_evaluation = model_client.import_model_evaluation(
        parent=versioned_model_name,
        model_evaluation=evaluation,
    )


    print(
        f"Imported evaluation: {imported_evaluation.name}",
        flush=True,
    )

    registered_model_name.metadata["model_resource_name"] = (
        model.resource_name
    )

    registered_model_name.metadata["version_id"] = (
        model.version_id
    )

@dsl.pipeline(
    name="facility-demand-forecasting",
    pipeline_root=PIPELINE_ROOT,
)
def forecasting_pipeline():

    load_task = load_training_data()

    train_task = train_lightgbm(
        training_data=load_task.outputs["training_data"],
    )

    evaluation_registration_task = evaluate_and_register(
        training_data=load_task.outputs["training_data"],
        trained_model=train_task.outputs["trained_model"],
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path="facility_forecasting_pipeline.yaml",
    )

    print("Pipeline compiled successfully.")
