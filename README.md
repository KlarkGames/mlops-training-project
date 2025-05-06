# ASR - Numbers Recognition in Russian

Kaggle competition to complete: https://www.kaggle.com/competitions/asr-numbers-recognition-in-russian/overview

# How to use

## Install dependencies:

```bash
pip install uv
uv sync
```

## Run Pipeline

Run local server:

```bash
mlflow server
```

Run pipeline:

```bash
dvc repro
```

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
