from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.11",
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