FROM python:3.12-bullseye as builder

RUN pip install --no-cache-dir uv

COPY pyproject.toml requirements.txt uv.lock ./

RUN uv sync --system

COPY . .

RUN useradd -m appuser
USER appuser

# CMD ["python", "-m", "your_package"]
