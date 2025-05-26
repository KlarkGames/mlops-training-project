from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
}

dag = DAG(
    "mlops_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

process_data = BashOperator(
    task_id="process_data",
    bash_command="uv run python -m src.data.data_download && uv run python -m src.data.data_prepare",
    dag=dag,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command=(
        "uv run python -m src.models.train "
        "--train-csv data/numbers/train.csv "
        "--val-csv data/numbers/dev.csv "
        "--train-audio-dir data/numbers/train "
        "--val-audio-dir data/numbers/dev "
        "--vocabulary-json data/numbers/vocabulary.json"
    ),
    dag=dag,
)

process_data >> train_model
