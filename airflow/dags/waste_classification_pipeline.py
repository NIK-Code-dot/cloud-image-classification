from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


def validate_dataset():
    import os

    train_dir = "ml/data/processed/train"
    val_dir = "ml/data/processed/val"
    test_dir = "ml/data/processed/test"

    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

    print("Dataset validation successful.")
    print("Train directory:", train_dir)
    print("Validation directory:", val_dir)
    print("Test directory:", test_dir)


with DAG(
    dag_id="waste_classification_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["waste-classification", "ml"],
) as dag:

    validate = PythonOperator(
        task_id="validate_dataset",
        python_callable=validate_dataset,
    )
