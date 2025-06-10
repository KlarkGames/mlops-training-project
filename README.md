# ASR - Numbers Recognition in Russian

Kaggle competition to complete: https://www.kaggle.com/competitions/asr-numbers-recognition-in-russian/overview


## Results

There are results from several runs on diffenert dataset formats:

Result of training with numbers representation (Audio -> 1234)
![alt text](doc/imgs/Numbers%20Dataset%20Training%20losses.png)

Result of training with numbers representation (Audio -> "тысяча двести тридцать четыре")
![alt text](doc/imgs/Words%20Dataset%20Training%20losses.png)

Result of training with numbers representation (Audio -> <1>|<200><30><4>)
![alt text](doc/imgs/Tokens%20Dataset%20Training%20losses.png)

Token representation allows us to achieve the best validation loss.

# Collaboration

To collaborate on this project please use [GitHub Workflow](https://docs.github.com/en/get-started/using-github/github-flow).

Please read [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification to enshure that your commits named properly.

## 🌀 Airflow Integration (Local Pipeline Orchestration)

This project supports orchestration using **Apache Airflow** with Docker Compose.

### ✅ Features

* Local Airflow deployment (via Docker)
* Pipeline with three steps: **Data Processing → Model Training → Model Testing**
* Integrated with **DVC** for data and artifact versioning

### 🔧 Setup

Make sure Docker and Docker Compose are installed.

```bash
docker-compose up -d
```

* Airflow UI will be available at: [http://localhost:8080](http://localhost:8080)
* Default credentials: `admin` / `admin`

If port `8080` is busy, change it in `docker-compose.yml`:

```yaml
ports:
  - "8081:8080"
```

### ▶️ Run the DAG

Once Airflow is running:

1. Go to the web UI.
2. Find DAG named `mlops_pipeline`.
3. Toggle it **on**, then click **Trigger DAG**.

The DAG contains:

* `process_data`: runs `data_download` and `data_prepare`
* `train_model`: runs model training

All steps are executed via `uv` for dependency consistency and tracked with `dvc`.

## Others Services URL's

- MLFlow: [http://localhost:9000](http://localhost:5000)
- MINIO: [http://localhost:7000](http://localhost:9000)

## FastAPI model service

The repository exposes a FastAPI application that wraps the existing training
pipeline located in `src/models` and `src/data`.  The service can download and
prepare data, perform inference using a trained Conformer model and report
metrics on the validation split.

### Run locally

```bash
uvicorn src.api.main:app --reload
```

To start the service with Docker Compose:

```bash
docker-compose up api
```

The model checkpoint and validation dataset paths are configured in
`service_config.yaml`.

### Endpoints

* `POST /download_data` &mdash; download the raw dataset from Kaggle.
* `POST /prepare_data` &mdash; generate dataset splits and vocabularies.
* `POST /predict` &mdash; return transcription for an uploaded audio file.
* `GET /metrics` &mdash; compute character error rate on the validation split.

### Example client

`python scripts/query_api.py` sends the validation set audio files to the API
and saves predictions to `results.json`.
