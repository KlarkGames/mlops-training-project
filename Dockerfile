FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/

RUN pip install uv dvc apache-airflow && \
    uv install

# Копируем код
COPY . /app


RUN dvc pull --quiet

ENTRYPOINT ["bash"]
