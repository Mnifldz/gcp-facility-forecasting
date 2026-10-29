import os
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import tempfile
from google.cloud import storage

app = FastAPI(title="Facility Demand Forecasting")


class PredictionRequest(BaseModel):
    instances: list[dict] = Field(min_length=1)


def load_model():
    model_uri = os.environ.get(
        "AIP_STORAGE_URI",
        "/app/model",
    )

    if model_uri.startswith("gs://"):
        # Parse gs://bucket-name/path/to/model-directory
        bucket_and_prefix = model_uri[5:]
        bucket_name, _, prefix = bucket_and_prefix.partition("/")

        blob_name = f"{prefix.rstrip('/')}/model.joblib"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_model_path = (
            Path(tempfile.gettempdir()) / "model.joblib"
        )

        blob.download_to_filename(str(local_model_path))

        print(
            f"Downloaded model from gs://{bucket_name}/{blob_name}",
            flush=True,
        )

    else:
        local_model_path = Path(model_uri) / "model.joblib"

    artifact = joblib.load(local_model_path)

    return (
        artifact["model"],
        artifact["features"],
        artifact["categories"],
    )


model = None
features = None
categories = None


@app.on_event("startup")
def startup():
    global model, features, categories

    model, features, categories = load_model()

    print("LightGBM model loaded successfully", flush=True)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.instances)

        X = df[features].copy()

        X["facility_id"] = pd.Categorical(
            X["facility_id"].astype(int),
            categories=categories,
        )

        predictions = model.predict(X)

        return {
            "predictions": predictions.tolist()
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
