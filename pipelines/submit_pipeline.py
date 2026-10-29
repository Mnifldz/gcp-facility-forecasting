from google.cloud import aiplatform

PROJECT_ID = "storage-analytics-lab"
REGION = "us-west2"

PIPELINE_ROOT = (
    "gs://storage-analytics-lab-vertex-artifacts/"
    "pipelines/facility-forecasting"
)

SERVICE_ACCOUNT = (
    "555144694235-compute@developer.gserviceaccount.com"
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket="gs://storage-analytics-lab-vertex-artifacts",
)

job = aiplatform.PipelineJob(
    display_name="facility-demand-forecasting",
    template_path="facility_forecasting_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

job.submit(
    service_account=SERVICE_ACCOUNT,
)

print("Pipeline submitted successfully")
print("Resource name:", job.resource_name)
