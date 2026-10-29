from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output


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
    from google.protobuf.struct_pb2 import Struct

    import joblib
    import pandas as pd

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
    )

    # --------------------------------------------------
    # 1. Load the trained model and validation dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 2. Configure model registration
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 3. Upload the model artifact to GCS
    # --------------------------------------------------

    run_id = uuid.uuid4().hex

    artifact_prefix = f"facility-demand/{run_id}"

    artifact_uri = (
        f"gs://{bucket_name}/{artifact_prefix}"
    )

    client = storage.Client(project=project)

    blob = client.bucket(bucket_name).blob(
        f"{artifact_prefix}/model.joblib"
    )

    blob.upload_from_filename(trained_model.path)

    print(f"Model artifact: {artifact_uri}", flush=True)

    # --------------------------------------------------
    # 4. Register a new model version
    # --------------------------------------------------

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
        upload_args["parent_model"] = parent_model
        upload_args["is_default_version"] = False
    else:
        upload_args["model_id"] = model_id

    registered_model = aiplatform.Model.upload(**upload_args)

    print(
        f"Registered model: {registered_model.resource_name}",
        flush=True,
    )

    print(
        f"Version ID: {registered_model.version_id}",
        flush=True,
    )

    # --------------------------------------------------
    # 5. Record the registered model version
    # --------------------------------------------------

    versioned_model_name = (
        f"{registered_model.resource_name}"
        f"@{registered_model.version_id}"
    )

    Path(registered_model_name.path).write_text(
        versioned_model_name
    )

    # --------------------------------------------------
    # 6. Import evaluation metrics into Model Registry
    # --------------------------------------------------

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

    imported_evaluation = model_client.import_model_evaluation(
        parent=versioned_model_name,
        model_evaluation=evaluation,
    )

    print(
        f"Imported evaluation: {imported_evaluation.name}",
        flush=True,
    )

    registered_model_name.metadata["model_resource_name"] = (
        registered_model.resource_name
    )

    registered_model_name.metadata["version_id"] = (
        registered_model.version_id
    )