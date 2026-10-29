from kfp import dsl, compiler

from pipelines.components.load_data import load_training_data
from pipelines.components.train import train_lightgbm
from pipelines.components.evaluate_register import evaluate_and_register


PIPELINE_ROOT = (
    "gs://storage-analytics-lab-vertex-artifacts/"
    "pipelines/facility-forecasting"
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

    evaluate_and_register(
        training_data=load_task.outputs["training_data"],
        trained_model=train_task.outputs["trained_model"],
    )


if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path="facility_forecasting_pipeline.yaml",
    )

    print("Pipeline compiled successfully.")